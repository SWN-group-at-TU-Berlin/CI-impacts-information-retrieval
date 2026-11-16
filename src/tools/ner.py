#!/usr/bin/env python
"""NER model (spaCy pipeline + CI patterns) and the NER tools of LLM 1 and the LLM-as-a-judge"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import re
from pathlib import Path

import pandas as pd
import spacy
from langchain_core.tools import tool

from src.settings import PROJECT_ROOT, init_logger, settings

logger = init_logger(__name__)

# spaCy pipeline: path set in configs/config.yaml (SPACY_MODEL_PIPELINE), CI patterns are part of the repository
SPACY_MODEL_PIPELINE_DIR = settings.SPACY_MODEL_PIPELINE
SPACY_BASE_MODEL = "en_core_web_trf"  # English transformer pipeline (RoBERTa), https://spacy.io/models/en#en_core_web_trf
NER_PATTERNS_DIR = PROJECT_ROOT / "ner_patterns.jsonl"


def load_ci_patterns() -> pd.DataFrame:
    """NER patterns of the CI types and their subgroups (needed for grouping the CI types of the LLM response)"""
    return pd.read_json(NER_PATTERNS_DIR / "patterns", lines=True)


def create_spacy_model_pipeline(output_dir: Path = SPACY_MODEL_PIPELINE_DIR, base_model: str = SPACY_BASE_MODEL) -> Path:
    """Create the spaCy pipeline of the NER model: base_model (downloaded if not installed) + merge_entities + merge_noun_chunks,
    saved to output_dir. The CI patterns (span_ruler) are added when the pipeline is loaded, see NERModel.
    """
    if not spacy.util.is_package(base_model):
        logger.info(f"Downloading spaCy model {base_model}")
        spacy.cli.download(base_model)
    nlp = spacy.load(base_model)
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("merge_noun_chunks")
    nlp.to_disk(output_dir)
    logger.info(f"Saved spaCy model pipeline to {output_dir}")
    return Path(output_dir)


# ## NER model


class NERModel:
    def __init__(self):
        self.nlp_model = self.initialize()

    def initialize(self):
        """Return initialized NER model"""

        dir_model_pipeline = SPACY_MODEL_PIPELINE_DIR
        dir_ci_patterns = NER_PATTERNS_DIR

        # LOAD spacy pipeline, created on first use
        if not Path(dir_model_pipeline).exists():
            logger.warning(f"spaCy model pipeline not found at {dir_model_pipeline}, creating it from {SPACY_BASE_MODEL}")
            create_spacy_model_pipeline(dir_model_pipeline)
        nlp = spacy.load(dir_model_pipeline)

        # add CI_TYPE patterns to spacy nlp model pipeline
        config = {"spans_key": None, "annotate_ents": True, "overwrite": False}
        ## see for more info: https://spacy.io/usage/rule-based-matching#entityruler
        ## NOTE EntityRuler is hidden inside .add_pipe()
        try:
            ruler = nlp.add_pipe("span_ruler", config=config)
            ruler.from_disk(dir_ci_patterns)
        except ValueError:
            print("SpanRuler already exists in pipeline.")
            ruler = nlp.get_pipe("span_ruler")
            ruler.from_disk(dir_ci_patterns)

        return nlp


_nlp_model = None


def get_nlp_model():
    """NER model, loaded once on first use"""
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = NERModel().nlp_model
    return _nlp_model


@tool
def get_infrastructure_entities(text: str) -> list[str] | None:
    """This tool detects infrastructure assets and services from text by using a spaCy model to conduct Named Entity Recognition (NER)"""

    # remove common misleading CI keywords
    # ("dam" - "damage.*", "train"- "training") as NER patterns not recognize stop-char "\b" for any reason
    # but keep when rest of entity is CI, eg, "damaged rails",
    regex = re.compile(r"\b[Dd]amage\w*|\b[Tt]raining\w*|\b[Cc]ommunication\w*")
    text_clean = re.sub(regex, "", text).replace("  ", " ")

    # get ci entities
    nlp_chunk = get_nlp_model()(text_clean)
    ci_entities = [ent for ent in nlp_chunk.ents if ent.label_ in ["CI_TYPE", "FAC"]]

    ## cleaning:
    # remove common errorneous CI keywords ("dam" - "damage.*", "train"- "training") as NER patterns not recognize stop-char "\b" for any reason
    regex = re.compile(r".*damage.*|.*training.*|\b[Cc]ommunication")
    ci_entities = [str(i) for i in ci_entities if not regex.match(str(i))]

    if ci_entities:
        return ci_entities

    return None

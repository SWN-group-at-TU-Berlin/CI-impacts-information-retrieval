from types import SimpleNamespace

import pandas as pd
import pytest
import spacy

import src.pipeline as pl
from src.tools import ner


@pytest.fixture
def fake_nlp_model(monkeypatch):
    """Mock of the spaCy model pipeline (spacy_model_pipeline/, not available e.g. on GitHub Actions):
    blank English model + the CI patterns of the project, used by the NER tool get_infrastructure_entities
    """
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("span_ruler", config={"spans_key": None, "annotate_ents": True, "overwrite": False})
    ruler.from_disk(ner.NER_PATTERNS_DIR)
    monkeypatch.setattr(ner, "get_nlp_model", lambda: nlp)
    return nlp


def test_prepare_chunk_text_restores_abbreviations_only_as_words():
    assert pl.prepare_chunk_text("roads, eg the A1, ie closed; the leg was hurt") == "roads, e.g. the A1, i.e. closed; the leg was hurt"


def test_get_citation():
    assert pl.get_citation("Koks et al 2022 - Brief communication") == "Koks 2022"


class FakeExtract:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def extract(self, text, **kwargs):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeJudge:
    def judge(self, df, text):
        return df.assign(judge_verdict="correct")


class FakeChunker:
    def __init__(self, texts):
        self.texts = texts

    def chunk(self, dl_doc):
        return [SimpleNamespace(text=t) for t in self.texts]


def records(**kwargs):
    return pd.DataFrame(
        [
            {
                "infrastructure_type": "roads",
                "location": "Berlin",
                "damage": "inundation",
                "societal_impact_amount": None,
                **kwargs,
            }
        ]
    )


def make_pipeline(monkeypatch, result, texts, tmp_path=None):
    monkeypatch.setattr(pl.le, "result_to_dataframe", lambda result: records())
    patterns = pd.DataFrame()
    monkeypatch.setattr(
        pl.pp,
        "group_ci_types",
        lambda df, col, col_grouped, patterns, **kw: df.assign(**{col_grouped: "Transportation"}),
    )
    return pl.CiImpactPipeline(
        FakeChunker(texts),
        FakeExtract(result),
        FakeJudge(),
        ci_patterns=patterns,
        output_dir=tmp_path,
        verbose=False,
    )


def test_pipeline_runs_ner_filter_llm_extractor_and_judge(monkeypatch, tmp_path, fake_nlp_model):
    pipeline = make_pipeline(
        monkeypatch,
        {"is_infrastructure_case": True},
        ["The roads in Berlin were flooded.", "The weather was nice."],
        tmp_path,
    )
    df = pipeline.run({"Koks 2022 - Brief communication": object()})
    assert pipeline.llm_extract.calls == 1  # second chunk has no infrastructure entity (NER pre-filter)
    assert df[["citation_id", "chunk_id", "infrastructure_group", "judge_verdict"]].values.tolist() == [["Koks 2022", 0, "Transportation", "correct"]]
    assert (tmp_path / "llm_chain_judge_Koks 2022.csv").exists()


def test_pipeline_records_llm_errors(monkeypatch, fake_nlp_model):
    pipeline = make_pipeline(monkeypatch, RuntimeError("server down"), ["The roads in Berlin were flooded."])
    df = pipeline.extract_document("Koks 2022 - Brief communication", object())
    assert df.empty
    assert pipeline.errors[0]["error"] == "RuntimeError: server down"

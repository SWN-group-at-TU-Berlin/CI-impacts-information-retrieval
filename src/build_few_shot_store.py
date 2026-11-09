#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the store of few-shot examples (src/few_shot.py) from the manually annotated table: split the publications
into pool and hold-out set, and save the records and embeddings of the pool to settings.PATH_FEW_SHOT.
usage (from the project root): uv run python -m src.build_few_shot_store
"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # makes "import src..." work from anywhere

import pandas as pd
import spacy

import src.few_shot as fs
import src.postprocess as pp
from src.settings import settings as s


def make_ner_fn():
    """Same NER and cleaning as in get_infrastructure_entities and extract_asset_information of notebooks/extract_ciimpacts.py"""
    nlp = spacy.load("./spacy_model_pipeline")
    ruler = nlp.add_pipe("span_ruler", config={"spans_key": None, "annotate_ents": True, "overwrite": False})
    ruler.from_disk(Path("./ner_patterns.jsonl"))
    ci_patterns = pd.read_json("./ner_patterns.jsonl/patterns", lines=True)

    def ner_fn(text: str) -> list[str]:
        text_clean = re.sub(r"\b[Dd]amage\w*|\b[Tt]raining\w*|\b[Cc]ommunication\w*", "", text).replace("  ", " ")
        regex = re.compile(r".*damage.*|.*training.*|\b[Cc]ommunication")
        entities = [str(ent) for ent in nlp(text_clean).ents if ent.label_ in ["CI_TYPE", "FAC"] and not regex.match(str(ent))]
        return pp.keep_ci_types(entities, ci_patterns)

    return ner_fn


def count_classes(records: list[dict]) -> Counter:
    return Counter(name for record in records for name in fs.split_classes(record))


df_table = pd.read_csv(s.PATH_VALID_DATA / s.FEW_SHOT_TABLE_FILENAME)
records, report = fs.build_records(df_table, make_ner_fn())
split = fs.make_split(records)

pool = [record for record in records if split[record["publication_id"]] == "pool"]
holdout = [record for record in records if split[record["publication_id"]] == "holdout"]
fs.assert_no_shared_sentences(pool, holdout)

embeddings = fs.make_encoder(s.FEW_SHOT_EMBEDDING_MODEL)([record["text"] for record in pool])
fs.save_store(pool, split, embeddings, s.PATH_FEW_SHOT)

print("Table:", report)
print("Pool publications:", sorted(p for p, side in split.items() if side == "pool"))
print(f"{'class':<15}{'pool':>6}{'total':>7}{'share':>8}")
n_pool, n_all = count_classes(pool), count_classes(records)
for name, total in n_all.items():
    print(f"{name:<15}{n_pool[name]:>6}{total:>7}{n_pool[name] / total:>8.2f}")
print("Unmapped damage in pool:", sum(case["damage"] is None for record in pool for case in record["direct"]),
      "of", sum(len(record["direct"]) for record in pool))
print(f"Saved {len(pool)} examples to {s.PATH_FEW_SHOT}")

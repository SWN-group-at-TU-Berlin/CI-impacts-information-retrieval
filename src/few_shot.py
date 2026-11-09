#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dynamic few-shot examples: per text chunk retrieve the most similar verified examples for the extraction prompts.
The examples come from the pool publications of the manually annotated table, all other publications are the hold-out set.
The store is built by src/build_few_shot_store.py"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import src.postprocess as pp


# table column which marks a cascading impact of the respective type
IMPACT_COLUMNS = {
    "economic": "economic_impact_type",
    "societal": "societal_impact_type",
    "secondary": "secondary_infrastructure_asset",
    "environmental": "environmental_impact_type",
}
IMPACT_NAMES = {
    "economic": "economic impact",
    "societal": "societal impact",
    "secondary": "secondary infrastructure impact",
    "environmental": "environmental impact",
}
# entries in the table which are not publications or impacts, i.e. values that slipped into the wrong column
MISALIGNED_PUBLICATIONS = ["unknown source", "not clear if this affected trnasport"]
MISALIGNED_SECONDARY_ASSETS = ["damaged"]

HEADER = "Further verified examples, similar to the TEXT. Use them only as guidance, do not extract information from them:\n\n"


def value_of(row, column: str) -> str | None:
    """Stripped cell value, None if the cell is empty"""
    value = row.get(column)
    if value is None or pd.isna(value) or str(value).strip().lower() in ["", "nan", "none"]:
        return None
    return str(value).strip()


def normalize_text(text: str) -> str:
    """Lowercase text without punctuation, used to compare reference sentences"""
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def impact_targets(rows: pd.DataFrame) -> list[str]:
    """Types of cascading impacts annotated in the rows"""
    return [
        task for task, column in IMPACT_COLUMNS.items()
        if any(value_of(row, column) for _, row in rows.iterrows())
    ]


def is_undone(df: pd.DataFrame) -> pd.Series:
    """Rows which are not fully annotated yet, same definition as in llm_evaluation.ipynb"""
    return df.astype(str).apply(lambda x: x.str.contains("xx")).any(axis=1)


def clean_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns the table with a publication and reference sentence (all rows), the same table without unusable rows, and the dropped counts"""
    df_raw = df.dropna(subset=["publication_id", "sentence_reference"]).copy()
    df_raw = df_raw[~df_raw["publication_id"].str.strip().isin(MISALIGNED_PUBLICATIONS)]
    df_raw["sentence_key"] = df_raw["sentence_reference"].map(normalize_text)
    df_raw = df_raw[df_raw["sentence_key"] != ""]

    undone = is_undone(df_raw.drop(columns="sentence_key"))
    has_impact = df_raw.apply(lambda row: any(value_of(row, c) for c in IMPACT_COLUMNS.values()), axis=1)
    no_source = df_raw.apply(
        lambda row: value_of(row, "ci1_type") is None and (value_of(row, "ci1_location") is None or value_of(row, "ci1_damage_group") is None),
        axis=1,
    ) & has_impact
    misaligned = df_raw["secondary_infrastructure_asset"].astype(str).str.strip().isin(MISALIGNED_SECONDARY_ASSETS)

    report = {
        "rows": len(df_raw),
        "dropped_undone": int(undone.sum()),
        "dropped_impact_without_source": int((no_source & ~undone).sum()),
        "dropped_misaligned": int((misaligned & ~undone & ~no_source).sum()),
    }
    return df_raw, df_raw[~(undone | no_source | misaligned)], report


def direct_cases(rows: pd.DataFrame) -> list[dict]:
    cases = [
        {
            "ci_asset": value_of(row, "ci1_type"),
            "ci_asset_amount": value_of(row, "ci1_amount"),
            "ci_asset_name": value_of(row, "ci1_name"),
            "location": value_of(row, "ci1_location"),
            "damage": pp.map_damage_to_category(value_of(row, "ci1_damage_group")),
        }
        for _, row in rows.iterrows() if value_of(row, "ci1_type")
    ]
    return unique(cases)


def impact_entries(rows: pd.DataFrame, task: str) -> list[dict]:
    entries = []
    for _, row in rows.iterrows():
        if not value_of(row, IMPACT_COLUMNS[task]):
            continue
        entry = {"source_asset": value_of(row, "ci1_type")}
        if task == "secondary":
            entry["source_damage"] = value_of(row, "ci1_damage_type")
            entry["source_location"] = value_of(row, "ci1_location")
            entry["target_asset"] = value_of(row, "secondary_infrastructure_asset")
        else:
            entry["source_location"] = value_of(row, "ci1_location")
            entry["impact_origin"] = value_of(row, f"{task}_impact_origin")
            entry["impact_type"] = value_of(row, f"{task}_impact_type")
            entry["amount"] = value_of(row, f"{task}_impact_amount")
        entries.append(entry)
    return unique(entries)


def unique(entries: list[dict]) -> list[dict]:
    """Remove duplicates: one case is repeated in the table for each of its impacts"""
    return [dict(items) for items in dict.fromkeys(tuple(entry.items()) for entry in entries)]


def build_records(df: pd.DataFrame, ner_fn=lambda text: []) -> tuple[list[dict], dict]:
    """One record per reference sentence, which may span several sentences. ner_fn returns the NER entities of a text.

    dropped_targets are impact types annotated only in unusable rows: such a text must not be used as an example without this impact.
    """
    df_raw, df_clean, report = clean_table(df)
    annotated = {key: impact_targets(rows) for key, rows in df_raw.groupby(["publication_id", "sentence_key"], sort=False)}

    records = []
    for key, rows in df_clean.groupby(["publication_id", "sentence_key"], sort=False):
        record = {
            "id": f"{key[0]}#{len(records)}",
            "publication_id": key[0],
            "text": rows["sentence_reference"].iloc[0].strip(),
            "direct": direct_cases(rows),
            "has_cascading_impacts": bool(annotated[key]),
        }
        for task in IMPACT_COLUMNS:
            record[task] = impact_entries(rows, task)
        record["dropped_targets"] = [task for task in annotated[key] if not record[task]]
        gold_entities = [case["ci_asset"] for case in record["direct"]]
        record["ner_entities"] = list(dict.fromkeys(list(ner_fn(record["text"]) or []) + gold_entities))
        records.append(record)

    report["records"] = len(records)
    return records, report


def split_classes(record: dict) -> list[str]:
    """Classes which should be equally represented in pool and hold-out set"""
    return ["cascade" if record["has_cascading_impacts"] else "no_cascade"] + [task for task in IMPACT_COLUMNS if record[task]]


def make_split(records: list[dict], pool_share: float = 1 / 3, seed: int = 42, n_iter: int = 200_000) -> dict[str, str]:
    """Assign whole publications to "pool" or "holdout", so that pool_share of the texts of each class end up in the pool.
    Searches random assignments and returns the one with the smallest maximum deviation."""
    publications = sorted({record["publication_id"] for record in records})
    classes = ["cascade", "no_cascade", *IMPACT_COLUMNS]
    counts = np.zeros((len(publications), len(classes)))
    for record in records:
        for name in split_classes(record):
            counts[publications.index(record["publication_id"]), classes.index(name)] += 1

    assignments = np.random.default_rng(seed).random((n_iter, len(publications))) < pool_share
    counts = counts[:, counts.sum(axis=0) > 0]  # a class without texts cannot be balanced
    shares = assignments @ counts / counts.sum(axis=0)
    best = assignments[np.abs(shares - pool_share).max(axis=1).argmin()]
    return {publication: "pool" if in_pool else "holdout" for publication, in_pool in zip(publications, best)}


def assert_no_shared_sentences(pool: list[dict], holdout: list[dict]) -> None:
    """No reference sentence of the pool may be part of the hold-out set, also not as a part of a longer reference"""
    pool_keys = {normalize_text(record["text"]) for record in pool}
    holdout_keys = {normalize_text(record["text"]) for record in holdout}
    shared = [(a, b) for a in pool_keys for b in holdout_keys if a in b or b in a]
    if shared:
        raise ValueError(f"{len(shared)} reference sentences are in both pool and hold-out set, e.g. {shared[0]}")


def split_into_windows(text: str) -> list[str]:
    """Windows of 1 and of 3 consecutive sentences, as the impacts of one case can spread over several sentences"""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) < 3:
        return sentences + [" ".join(sentences)] if len(sentences) > 1 else sentences or [text]
    return sentences + [" ".join(sentences[i:i + 3]) for i in range(len(sentences) - 2)]


def format_none(value) -> str:
    return "None" if value is None else f'"{value}"'


def format_ci_damages(record: dict) -> str:
    return "; ".join(
        f'ci_asset={format_none(case["ci_asset"])}, location={format_none(case["location"])}, damage={format_none(case["damage"])}'
        for case in record["direct"]
    ) or "None"


def render_direct(record: dict) -> str:
    cases = ", ".join(
        "InfrastructureCase(" + ", ".join(f"{field}={format_none(value)}" for field, value in case.items()) + ")"
        for case in record["direct"]
    )
    return f"Example Input:\ntext: {record['text']}\nINFRASTRUCTURE ENTITIES DETECTED BY NER: {record['ner_entities']}\n\nExample Output:\n[{cases}]"


def render_detect(record: dict) -> str:
    effects = [
        f"- {entry['source_asset']} → {entry.get('impact_type') or entry.get('target_asset')}"
        for task in IMPACT_COLUMNS for entry in record[task]
    ]
    return f"Text: {record['text']}\n" + ("\n".join(effects) or "- no cascading impact")


def render_impacts(record: dict, task: str) -> str:
    head = f"TEXT: {record['text']}\nCI_DAMAGES: {format_ci_damages(record)}\n"
    if not record[task]:
        return head + f"No {IMPACT_NAMES[task]} is mentioned: impacts = []"
    return head + "\n".join(f"{field}: {value}" for entry in record[task] for field, value in entry.items())


def render_example(record: dict, task: str) -> str:
    if task == "direct":
        return render_direct(record)
    if task == "detect":
        return render_detect(record)
    return render_impacts(record, task)


def render(selected: dict[str, list]) -> dict[str, str]:
    """Text of the selected examples per task, in the format of the static examples of the respective prompt"""
    return {
        task: HEADER + "\n\n".join(render_example(record, task) for record, _ in examples) + "\n\n" if examples else ""
        for task, examples in selected.items()
    }


def log_rows(selected: dict[str, list], citation: str, chunk_id: int) -> list[dict]:
    """Selected examples as rows for the log of the run"""
    return [
        {"citation_id": citation, "chunk_id": chunk_id, "task": task, "example_id": record["id"], "similarity": round(score, 3)}
        for task, examples in selected.items() for record, score in examples
    ]


def is_positive(record: dict, task: str) -> bool:
    if task == "direct":
        return bool(record["direct"])
    if task == "detect":
        return any(record[t] for t in IMPACT_COLUMNS)
    return bool(record[task])


def is_negative(record: dict, task: str) -> bool:
    if task == "direct":
        return False
    if task == "detect":
        return not record["has_cascading_impacts"]
    # a cascade of a different kind
    return record["has_cascading_impacts"] and not record[task] and task not in record["dropped_targets"] and any(record[t] for t in IMPACT_COLUMNS)


class FewShotStore:
    """Verified examples with their normalized embeddings. encode returns normalized embeddings for a list of texts."""

    def __init__(self, records: list[dict], embeddings: np.ndarray, encode, counts: dict, min_sim: float):
        self.records = records
        self.embeddings = embeddings
        self.encode = encode
        self.counts = counts
        self.min_sim = min_sim

    @classmethod
    def load(cls, directory: Path, model_name: str, counts: dict, min_sim: float):
        records = [json.loads(line) for line in open(Path(directory) / "examples.jsonl", encoding="utf-8")]
        embeddings = np.load(Path(directory) / "embeddings.npz")["embeddings"]
        return cls(records, embeddings, make_encoder(model_name), counts, min_sim)

    def _pick(self, task: str, order: np.ndarray, scores: np.ndarray) -> list:
        n_positive, n_negative = self.counts[task]
        positives, negatives, publications = [], [], set()
        for i in order:
            record, score = self.records[i], float(scores[i])
            if score < self.min_sim:
                break
            if record["publication_id"] in publications:
                continue
            if is_positive(record, task) and len(positives) < n_positive:
                positives.append((record, score))
            elif is_negative(record, task) and len(negatives) < n_negative:
                negatives.append((record, score))
            else:
                continue
            publications.add(record["publication_id"])
        return positives + negatives

    def select(self, text: str) -> dict[str, list]:
        """Per task the most similar examples above min_sim as (record, similarity), at most one per publication"""
        scores = (self.encode(split_into_windows(text)) @ self.embeddings.T).max(axis=0)
        order = np.argsort(-scores)
        return {task: self._pick(task, order, scores) for task in self.counts}

    def select_all(self, text: str) -> dict[str, str]:
        return render(self.select(text))


def make_encoder(model_name: str):
    """Encoder running on the CPU, as the GPUs are fully allocated by the vLLM servers"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    return lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def save_store(records: list[dict], split: dict[str, str], embeddings: np.ndarray, directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "examples.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    np.savez(directory / "embeddings.npz", embeddings=embeddings)
    pd.DataFrame(split.items(), columns=["publication_id", "split"]).to_csv(directory / "split.csv", index=False)


def holdout_publications(directory: Path) -> list[str]:
    df_split = pd.read_csv(Path(directory) / "split.csv")
    return df_split.loc[df_split["split"] == "holdout", "publication_id"].tolist()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluation utilities"""

import os
import re

import numpy as np
import pandas as pd



def true_positives(df_similarity: pd.DataFrame, df_valid, df_valid_col, df_pred, df_pred_col, similarity_threshold: float) -> int:
    tps = df_similarity.loc[df_similarity["impact_sim_identical"] >= similarity_threshold]
    tps_validmergedpred = df_valid.merge(df_pred, left_on=[df_valid_col], right_on=[df_pred_col], how="inner")
    assert len(tps) == len(tps_validmergedpred)
    
    return len(tps)


def false_negatives(df_valid, df_pred, df_valid_col, df_pred_col) -> int:
    tps = df_valid.merge(df_pred, left_on=[df_valid_col], right_on=[df_pred_col], how="inner")
    fps_len = len(df_pred[df_pred_col]) - len(tps)
    fns_len = len(df_valid[df_valid_col]) - len(tps)
    return fns_len


def false_positives(df_valid, df_pred, df_valid_col, df_pred_col) -> int:
    tps = df_valid.merge(df_pred, left_on=[df_valid_col], right_on=[df_pred_col], how="inner")
    fps_len = len(df_pred[df_pred_col]) - len(tps)
    return fps_len


##########################  PARAGRAPH-WISE EVALUATION
# Each validation record (sentence) is assigned to the prediction paragraph (chunk) it matches best.
# Within each paragraph, predictions and validation records are matched 1:1, and TPs, FPs, FNs are
# counted per paragraph, so every prediction and every validation record is counted exactly once.

# entity: (column in df_pred_valid_all / df_pred, column in df_pred_valid_all / df_valid)
PARAGRAPH_ENTITIES = {
    "ci": {"pred": "ci_group_pred", "valid": "ci_group_valid", "df_pred": "infrastructure_group", "df_valid": "ci1_group", "smlrty": "ci_smlrty"},
    "damage": {"pred": "damage_pred", "valid": "damage_valid", "df_pred": "damage", "df_valid": "ci1_damage", "smlrty": "dam_smlrty"},
    "location": {"pred": "location_pred", "valid": "location_valid", "df_pred": "location", "df_valid": "ci1_location", "smlrty": "loc_smlrty_norm_pr"},
}
PARAGRAPH_KEY = ["citation_id", "chunk_id"]


def assign_valid_to_paragraphs(df_pred_valid_all: pd.DataFrame) -> pd.DataFrame:
    """For each validation record keep the prediction paragraph with the highest text similarity
    to its sentence. Returns one row per id_valid with columns [id_valid, citation_id, chunk_id]."""
    best = df_pred_valid_all.loc[df_pred_valid_all.groupby("id_valid")["text_similarity"].idxmax()]
    return best[["id_valid", "citation_id", "chunk_id_pred"]].rename(columns={"chunk_id_pred": "chunk_id"}).reset_index(drop=True)


def select_paragraph_pairs(df_pred_valid_all: pd.DataFrame, df_valid_paragraphs: pd.DataFrame) -> pd.DataFrame:
    """Keep only pred-valid pairs where the prediction comes from the paragraph assigned to the validation record."""
    return df_pred_valid_all.merge(
        df_valid_paragraphs.rename(columns={"chunk_id": "chunk_id_pred"}),
        on=["id_valid", "citation_id", "chunk_id_pred"],
        how="inner",
    )


def calc_pair_similarities(df_pairs: pd.DataFrame, damage_similarity, location_similarity) -> pd.DataFrame:
    """Add similarity columns for CI (identical subgroup), damage and location to each pred-valid pair.

    damage_similarity, location_similarity: callables (pred: str, valid: str) -> float in [0, 1];
        only called when both values are strings, otherwise the similarity is 0
    """
    def _smlrty(pred, valid, func):
        return func(pred, valid) if isinstance(pred, str) and isinstance(valid, str) else 0.0

    df_pairs = df_pairs.copy()
    ci = PARAGRAPH_ENTITIES["ci"]
    df_pairs[ci["smlrty"]] = [
        float(isinstance(p, str) and p == v) for p, v in zip(df_pairs[ci["pred"]], df_pairs[ci["valid"]])
    ]
    for entity, func in [("damage", damage_similarity), ("location", location_similarity)]:
        e = PARAGRAPH_ENTITIES[entity]
        df_pairs[e["smlrty"]] = [_smlrty(p, v, func) for p, v in zip(df_pairs[e["pred"]], df_pairs[e["valid"]])]

    return df_pairs


def match_pairs_one_to_one(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """Greedy 1:1 matching of predictions and validation records within each paragraph:
    pairs are taken in order of highest similarity (CI first, then damage, then location),
    skipping pairs whose prediction or validation record is already matched."""
    smlrty_cols = [e["smlrty"] for e in PARAGRAPH_ENTITIES.values()]
    df_sorted = df_pairs.sort_values(smlrty_cols, ascending=False, kind="stable")

    matched_idx, used_pred, used_valid = [], set(), set()
    for idx, id_pred, id_valid in zip(df_sorted.index, df_sorted["id_pred"], df_sorted["id_valid"]):
        if id_pred in used_pred or id_valid in used_valid:
            continue
        matched_idx.append(idx)
        used_pred.add(id_pred)
        used_valid.add(id_valid)

    return df_pairs.loc[matched_idx]


def count_paragraph_scores(
    df_matched: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_valid_paragraphs: pd.DataFrame,
    thresholds: dict,
) -> pd.DataFrame:
    """Count TPs, FPs, FNs and compute precision, recall, F1 for each paragraph and entity.

    df_matched: 1:1 matched pred-valid pairs incl. similarity columns
    df_pred: all predictions with columns citation_id, chunk_id, id_pred and the entity columns
    df_valid: all validation records with columns publication_id, id_valid and the entity columns
    df_valid_paragraphs: output of assign_valid_to_paragraphs()
    thresholds: minimum similarity for a TP per entity, e.g. {"ci": 1.0, "damage": 0.7, "location": 0.7}

    Validation records that match no paragraph are counted as FNs under chunk_id = NaN of their document.
    Precision / recall are NaN for a paragraph when they are undefined (no predictions / no validation records).
    """
    preds = df_pred[PARAGRAPH_KEY + ["id_pred"] + [e["df_pred"] for e in PARAGRAPH_ENTITIES.values()]]
    valids = df_valid.merge(df_valid_paragraphs, on="id_valid", how="left")
    valids["citation_id"] = valids["citation_id"].fillna(valids["publication_id"])
    matched = df_matched.merge(df_valid_paragraphs[["id_valid", "chunk_id"]], on="id_valid", how="left")

    records = []
    for entity, e in PARAGRAPH_ENTITIES.items():
        tp_mask = (
            matched[e["pred"]].notna()
            & matched[e["valid"]].notna()
            & (matched[e["smlrty"]] >= thresholds[entity])
        )
        tps = matched[tp_mask].groupby(PARAGRAPH_KEY, dropna=False).size()
        n_pred = preds[preds[e["df_pred"]].notna()].groupby(PARAGRAPH_KEY, dropna=False).size()
        n_valid = valids[valids[e["df_valid"]].notna()].groupby(PARAGRAPH_KEY, dropna=False).size()

        counts = pd.concat({"tps": tps, "n_pred": n_pred, "n_valid": n_valid}, axis=1).fillna(0).astype(int)
        counts["fps"] = counts["n_pred"] - counts["tps"]
        counts["fns"] = counts["n_valid"] - counts["tps"]
        counts["impact"] = entity
        records.append(counts.reset_index())

    df_scores = pd.concat(records, ignore_index=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        df_scores["precision"] = df_scores["tps"] / df_scores["n_pred"].replace(0, np.nan)
        df_scores["recall"] = df_scores["tps"] / df_scores["n_valid"].replace(0, np.nan)
        df_scores["f1"] = 2 * df_scores["precision"] * df_scores["recall"] / (df_scores["precision"] + df_scores["recall"])
    df_scores.loc[(df_scores["precision"] == 0) & (df_scores["recall"] == 0), "f1"] = 0.0

    return df_scores[["impact"] + PARAGRAPH_KEY + ["tps", "fps", "fns", "n_pred", "n_valid", "precision", "recall", "f1"]]


def fuzzy_match_counts(pred: list, valid: list, scorer, threshold: float) -> tuple[int, int, int, list]:
    """Count TPs, FPs, FNs between a list of predicted and validated entities using fuzzy 1:1 matching.

    All pred-valid pairs are scored with scorer (e.g. fuzz.partial_ratio, 0-100) and matched greedily,
    highest score first; a pair is a TP when its score >= threshold and neither entity is matched yet.
    Returns (tps, fps, fns, matched pairs as (pred, valid, score)).
    """
    scored = sorted(
        ((scorer(str(p), str(v)), i, j) for i, p in enumerate(pred) for j, v in enumerate(valid)),
        reverse=True,
    )
    used_pred, used_valid, matches = set(), set(), []
    for score, i, j in scored:
        if score < threshold:
            break
        if i in used_pred or j in used_valid:
            continue
        used_pred.add(i)
        used_valid.add(j)
        matches.append((pred[i], valid[j], score))

    tps = len(matches)
    return tps, len(pred) - tps, len(valid) - tps, matches


def normalize_numbers(text: str) -> str:
    """Write all numbers in a string in one format, e.g. "20 000", "20,000", "20.000", "20'000" -> "20000",
    "2,5" -> "2.5", "1.000,5" -> "1000.5", "2.50" -> "2.5".

    Thousands separators (space, non-breaking space, apostrophe, comma, dot) are only removed when they
    are used consistently and followed by groups of exactly three digits, so "1.500" is read as 1500, not 1.5.
    """
    text = str(text)

    # 1. remove thousands separators (same separator between all 3-digit groups)
    text = re.sub(
        r"(?<![\d.,])\d{1,3}(?:([ \u00a0\u202f',.])\d{3})(?:\1\d{3})*(?!\d)",
        lambda m: re.sub(r"[ \u00a0\u202f',.]", "", m.group(0)),
        text,
    )
    # 2. decimal comma -> decimal point
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)

    # 3. one notation per value, e.g. "2.50" -> "2.5", "3.0" -> "3"
    def _canonical(m):
        value = float(m.group(0))
        return str(int(value)) if value.is_integer() else repr(value)

    return re.sub(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", _canonical, text)


def extract_numbers(text: str) -> list[float]:
    """Extract all numbers from a string after normalize_numbers(), e.g. "20 000 km and 2,5 bridges" -> [2.5, 20000.0]."""
    return sorted(float(n) for n in re.findall(r"\d+(?:\.\d+)?", normalize_numbers(text)))


def numbers_must_match(scorer):
    """Wrap a string scorer (e.g. fuzz.partial_ratio) so that a pair scores 0 unless both strings
    contain the same numbers; if they do (or neither contains a number), the original scorer is applied
    on the strings with normalized numbers.
    Repeated numbers are ignored, e.g. "100 to 200, between 100 to 200 households" matches "100 to 200"."""
    def _scorer(pred: str, valid: str) -> float:
        if set(extract_numbers(pred)) != set(extract_numbers(valid)):
            return 0
        return scorer(normalize_numbers(pred), normalize_numbers(valid))
    return _scorer


MISSING_IMPACT_VALUES = ["", "none", "nan"]


def is_missing_impact(value, pair_sep: str = " - ") -> bool:
    """True when an impact value is missing (NaN, None, "", "none", "nan" - any case), also for combined
    impact pairs such as "roads - nan" (e.g. cigroup_location) where one of the parts is missing."""
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return True
    return any(part.strip().lower() in MISSING_IMPACT_VALUES for part in str(value).split(pair_sep))


def drop_missing_impacts(values: list) -> list:
    """Remove missing impact values (see is_missing_impact), i.e. evaluate only on cases with an impact value."""
    return [x for x in values if not is_missing_impact(x)]


def calc_scores(tps: int, fps: int, fns: int, zero_division: float = 0.0) -> tuple[float, float, float]:
    """Precision, recall, F1 from TPs, FPs, FNs; returns zero_division for a score whose denominator is 0
    (e.g. precision when nothing was predicted)."""
    precision = tps / (tps + fps) if tps + fps > 0 else zero_division
    recall = tps / (tps + fns) if tps + fns > 0 else zero_division
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else zero_division
    return precision, recall, f1

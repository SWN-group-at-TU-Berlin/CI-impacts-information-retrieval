#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM-as-a-judge: check extracted CI impact records against the text chunk they were extracted from"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import json
import re
import typing

import numpy as np
import pandas as pd


JUDGE_VERDICTS = ["correct", "adapted", "false"]
NOT_JUDGED = "not_judged"  # judge gave no (valid) verdict for this record, record is kept unchanged

JUDGE_COLUMNS = [
    "judge_verdict",           # correct | adapted | false | not_judged
    "judge_adapted_fields",    # JSON: {field: {"old": ..., "new": ...}} for adapted records
    "judge_reason",            # short explanation of the judge
    "judge_evidence",          # sentence(s) of the chunk text the verdict is based on
    "judge_evidence_in_text",  # True if the evidence is found verbatim in the chunk text
    "judge_adaptation_in_text",  # True if all adapted (new) values are found in the chunk text, None if nothing adapted
]


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in ["", "nan", "none"]


def _to_json_value(value):
    if _is_missing(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def literal_values(annotation) -> list[str]:
    """All string values of a (nested) Literal type annotation, e.g. Literal[None, "damaged", "flooded"] | None -> ["damaged", "flooded"]."""
    args = typing.get_args(annotation)
    if not args:
        return [annotation] if isinstance(annotation, str) else []
    return [v for arg in args for v in literal_values(arg)]


def records_to_judge_payload(df: pd.DataFrame, fields: list[str]) -> str:
    """Serialize the records of df (only the given fields) as JSON list, each with a positional record_id."""
    fields = [f for f in fields if f in df.columns]
    records = [
        {"record_id": record_id, **{f: _to_json_value(row[f]) for f in fields}}
        for record_id, (_, row) in enumerate(df.iterrows())
    ]
    return json.dumps(records, ensure_ascii=False, indent=1)


def parse_judge_response(text: str) -> list[dict]:
    """Extract the list of judgements from the raw LLM answer (may contain reasoning, markdown code fences
    or a wrapping object such as {"records": [...]}). Raises ValueError if no JSON list is found."""
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)

    start, end = text.find("["), text.rfind("]")
    candidates = [text[start:end + 1]] if start != -1 and end > start else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed = next((v for v in parsed.values() if isinstance(v, list)), [parsed])
        if isinstance(parsed, list):
            return [j for j in parsed if isinstance(j, dict)]

    raise ValueError(f"No JSON list of judgements found in LLM response: {text[:500]}")


_LEADING_DETERMINERS = re.compile(
    r"^(?:(?:the|a|an|some|several|many|numerous|multiple|various|all|most|few|a few|no|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|dozens of|hundreds of|thousands of|\d[\d.,]*)\s+)+",
    flags=re.IGNORECASE,
)


def strip_determiners(value) -> str:
    """Remove leading articles, quantifiers and numbers, e.g. "Two bridges" -> "bridges", "the A-7 motorway" -> "A-7 motorway"."""
    return _LEADING_DETERMINERS.sub("", _normalize_text(value)).strip()


def _is_allowed(value, allowed: list) -> bool:
    """True if value is an allowed value, or a comma-separated combination of allowed values
    (cascading impact columns can hold several values, e.g. "repair_recovery_cost, downstream_economic_consequence")."""
    if value in allowed:
        return True
    return all(part.strip() in allowed for part in str(value).split(","))


def apply_judgements(
    df: pd.DataFrame,
    judgements: list[dict],
    fields: list[str],
    chunk_text: str | None = None,
    allowed_values: dict[str, list] | None = None,
    protect_values_in_text: bool = True,
    normalize_values: dict | None = None,
) -> pd.DataFrame:
    """Add the judge verdict to each record of df and apply the adaptations of "adapted" records.

    No record is removed: records without a valid judgement get verdict "not_judged", "false" records
    are kept unchanged. Only fields listed in `fields` can be adapted; adaptations to other fields, or to
    values not in allowed_values (e.g. damage categories), are ignored. With protect_values_in_text, an adaptation
    that sets a field to null is ignored when the current value is found verbatim in chunk_text (the judge must
    not remove information stated in the text). normalize_values maps fields to a function (e.g. strip_determiners);
    an adaptation is ignored if old and new value are equal after this function, e.g. "bridges" -> "Two bridges".
    An "adapted" record without any effective change is set to "correct".
    The record_id of a judgement refers to the position of the record in df (see records_to_judge_payload).
    """
    df = df.copy()
    allowed_values = allowed_values or {}
    normalize_values = normalize_values or {}
    text_norm = _normalize_text(chunk_text) if chunk_text is not None else None

    results = [
        {"judge_verdict": NOT_JUDGED, "judge_adapted_fields": None, "judge_reason": None,
         "judge_evidence": None, "judge_evidence_in_text": None, "judge_adaptation_in_text": None}
        for _ in range(len(df))
    ]
    judged = set()

    for judgement in judgements:
        try:
            record_id = int(judgement.get("record_id"))
        except (TypeError, ValueError):
            continue
        if not 0 <= record_id < len(df) or record_id in judged:
            continue

        verdict = str(judgement.get("verdict", "")).strip().lower()
        if verdict not in JUDGE_VERDICTS:
            continue
        judged.add(record_id)

        result = results[record_id]
        result["judge_reason"] = judgement.get("reason")
        evidence = judgement.get("evidence")
        result["judge_evidence"] = evidence
        if text_norm is not None and not _is_missing(evidence):
            result["judge_evidence_in_text"] = _normalize_text(evidence) in text_norm

        changes = {}
        if verdict == "adapted":
            adapted_fields = judgement.get("adapted_fields") or {}
            if not isinstance(adapted_fields, dict):
                adapted_fields = {}
            for field, new in adapted_fields.items():
                if field not in fields or field not in df.columns:
                    continue
                new = None if _is_missing(new) else new
                if new is not None and field in allowed_values and not _is_allowed(new, allowed_values[field]):
                    continue
                old = _to_json_value(df.iat[record_id, df.columns.get_loc(field)])
                if new == old:
                    continue
                if (
                    field in normalize_values and new is not None and old is not None
                    and normalize_values[field](new) == normalize_values[field](old)
                ):
                    continue
                if (
                    new is None and protect_values_in_text and text_norm is not None
                    and old is not None and _normalize_text(old) in text_norm
                ):
                    continue
                changes[field] = {"old": old, "new": new}
                df.iat[record_id, df.columns.get_loc(field)] = new if new is not None else np.nan

            if not changes:
                verdict = "correct"
            else:
                result["judge_adapted_fields"] = json.dumps(changes, ensure_ascii=False)
                if text_norm is not None:
                    new_values = [str(c["new"]) for c in changes.values() if c["new"] is not None]
                    result["judge_adaptation_in_text"] = all(_normalize_text(v) in text_norm for v in new_values)

        result["judge_verdict"] = verdict

    for column in JUDGE_COLUMNS:
        df[column] = [r[column] for r in results]

    return df


def format_judge_report(df_judged: pd.DataFrame, verdicts: str | tuple[str, ...] = ("adapted", "false")) -> str:
    """Readable summary of the judge verdicts: counts per verdict, and for each record with one of the given
    verdicts (e.g. "adapted" or ("adapted", "false")) its CI type and location, the changed fields (old -> new)
    and the reason of the judge."""
    if isinstance(verdicts, str):
        verdicts = (verdicts,)
    counts = df_judged["judge_verdict"].value_counts().to_dict()
    lines = [f"LLM-as-a-judge verdicts: {counts}"]

    for verdict in verdicts:
        df_verdict = df_judged[df_judged["judge_verdict"] == verdict]
        if df_verdict.empty:
            continue
        lines.append(f"  {verdict.upper()} records ({len(df_verdict)}):")

        for idx, row in df_verdict.iterrows():
            # describe the record by its values before adaptation
            changes = json.loads(row["judge_adapted_fields"]) if isinstance(row.get("judge_adapted_fields"), str) else {}
            label = [
                str(changes[f]["old"] if f in changes else _to_json_value(row.get(f)))
                for f in ["infrastructure_type", "location", "damage"]
            ]
            lines.append(f"    record {idx}: {' | '.join(label)}")
            for field, change in changes.items():
                lines.append(f"      {field}: {change['old']!r} -> {change['new']!r}")
            if not _is_missing(row.get("judge_reason")):
                lines.append(f"      reason: {row['judge_reason']}")

    return "\n".join(lines)

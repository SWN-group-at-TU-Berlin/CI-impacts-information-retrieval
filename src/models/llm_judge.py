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

from src.settings import init_logger


logger = init_logger(__name__)


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
    """True if value is an allowed value, or a combination of allowed values separated by "," or ";"
    (e.g. "repair_recovery_cost, downstream_economic_consequence" or damage categories "inundation; disruption")."""
    if value in allowed:
        return True
    return all(part.strip() in allowed for part in re.split(r"[,;]", str(value)))


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


class LlmJudge:
    """LLM-as-a-judge (step 3): check each extracted record of a chunk on its plausibility, based ONLY on the chunk text.
    The judge can accept ("correct") or adapt ("adapted") a record, or mark it as "false" - records are never removed.
    It uses the same field definitions (pydantic schemas of LLM 1), keyword lists (Literal) and NER entities as LLM 1.

    field_schema: record column -> (pydantic model, field name) whose definition is given to the judge
    extra_rules: record column -> additional rules on top of the field definition
    prompt: ChatPromptTemplate with the variables text, ner_entities, records and field_rules
    ner_fn: text -> list of NER entities (guidance for infrastructure_type and secondary_infrastructure_asset)

    usage:
        llm_judge = LlmJudge(s.JUDGE_MODEL, s.JUDGE_BASE_URL, s.VLLM_API_KEY, prompt=judge_prompt, field_schema=judge_field_schema)
        df_judged = llm_judge.judge(df_records, chunk_text)
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        prompt,
        field_schema: dict,
        extra_rules: dict | None = None,
        ner_fn=None,
        batch_size: int = 5,  # max. number of records judged in one LLM call (keeps reasoning + answer within max_tokens)
        max_consecutive_failures: int = 3,  # afterwards the judge is skipped for the rest of the run (records get "not_judged")
        max_retries: int = 1,
        retry_wait: int = 30,
        normalize_values: dict | None = None,
        protect_values_in_text: bool = True,
        temperature: float = 0,
        seed: int = 42,
        timeout: int = 300,
        max_tokens: int = 8000,  # the judge (reasoning model) needs tokens for reasoning + JSON answer; vLLM --max-model-len must be >= prompt + max_tokens
        llm=None,
    ):
        if llm is None:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key,
                             temperature=temperature, seed=seed, timeout=timeout, max_tokens=max_tokens)
        self.llm = llm
        self.prompt = prompt
        self.field_schema = field_schema
        self.fields = list(field_schema)  # record fields the judge checks and may adapt
        self.extra_rules = extra_rules or {}
        self.ner_fn = ner_fn or (lambda text: [])
        self.batch_size = batch_size
        self.max_consecutive_failures = max_consecutive_failures
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        # default: ignore changes that only add/remove articles, quantifiers or numbers, e.g. "bridges" -> "Two bridges"
        self.normalize_values = normalize_values if normalize_values is not None else {
            "infrastructure_type": strip_determiners, "secondary_infrastructure_asset": strip_determiners,
        }
        self.protect_values_in_text = protect_values_in_text

        # fields with a fixed keyword list in the schema (Literal), e.g. damage, economic_impact_origin -> judge may only use these keywords
        self.allowed_values = {
            column: literal_values(model.model_fields[name].annotation)
            for column, (model, name) in field_schema.items()
            if literal_values(model.model_fields[name].annotation)
        }
        self.field_rules = self.build_field_rules()
        self.reset()

    def reset(self) -> None:
        """Reset the failure counter, e.g. after the model endpoint is reachable again"""
        self.consecutive_failures = 0
        self.last_error = None  # shown when the judge is skipped, as the per-attempt messages may be far up in the output

    def build_field_rules(self) -> str:
        """Definition, allowed keywords and extra rules for each judged field, as text for the judge prompt."""
        blocks = []
        for column, (model, name) in self.field_schema.items():
            lines = [f'"{column}":', f"- Definition: {model.model_fields[name].description}"]
            if column in self.allowed_values:
                lines.append(f"- Allowed values (or null): {', '.join(self.allowed_values[column])}")
            else:
                lines.append("- Free text: the examples in the definition are NOT a closed list of allowed values. Keep the wording of the TEXT.")
            lines += [rule if rule.startswith("-") else f"- {rule}" for rule in self.extra_rules.get(column, [])]
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @property
    def is_skipped(self) -> bool:
        return self.consecutive_failures >= self.max_consecutive_failures

    def call(self, df_batch: pd.DataFrame, chunk_text: str, ner_entities: list[str]) -> list[dict] | None:
        """Ask the LLM to judge the records of df_batch. Returns the parsed judgements, or None if all attempts failed."""
        import time
        import openai

        messages = self.prompt.format_messages(
            text=chunk_text,
            ner_entities=json.dumps(ner_entities, ensure_ascii=False),
            records=records_to_judge_payload(df_batch, self.fields),
            field_rules=self.field_rules,
        )
        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                finish_reason = response.response_metadata.get("finish_reason")
                if not response.content.strip() and finish_reason == "length":
                    # reasoning models (e.g. gemma-4 in thinking mode) can use up all tokens for reasoning before writing the answer
                    usage = response.response_metadata.get("token_usage", {}) or {}
                    raise ValueError(
                        f"answer cut off before any JSON was written (finish_reason=length, completion_tokens={usage.get('completion_tokens')}). "
                        "Increase max_tokens of the judge or decrease batch_size."
                    )
                judgements = parse_judge_response(response.content)
                self.consecutive_failures = 0
                return judgements
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                print(f"! LLM-as-a-judge failed (attempt {attempt + 1}/{self.max_retries + 1}): {self.last_error}")
                # model not found / wrong api key: retrying does not help
                if isinstance(e, (openai.NotFoundError, openai.AuthenticationError, openai.PermissionDeniedError)):
                    break
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait)

        self.consecutive_failures += 1
        return None

    def judge(self, df_records: pd.DataFrame, chunk_text: str) -> pd.DataFrame:
        """Let the LLM judge each record of df_records against chunk_text (in batches of batch_size records).
        Returns df_records (same rows) with adapted values and the judge columns, see JUDGE_COLUMNS."""
        # same NER tool as for LLM 1, as guidance for infrastructure_type and secondary_infrastructure_asset
        ner_entities = self.ner_fn(chunk_text) or []

        df_batches_judged = []
        for batch_start in range(0, max(len(df_records), 1), self.batch_size):
            df_batch = df_records.iloc[batch_start:batch_start + self.batch_size]

            judgements = []
            if self.is_skipped:
                print(f"! LLM-as-a-judge skipped: failed {self.consecutive_failures} times in a row. Call .reset() to reset.\n  Last error: {self.last_error}")
            elif len(df_batch):
                judgements = self.call(df_batch, chunk_text, ner_entities) or []

            # record_ids in the judgements refer to the positions within this batch
            df_batches_judged.append(apply_judgements(
                df_batch, judgements, self.fields,
                chunk_text=chunk_text,
                allowed_values=self.allowed_values,
                normalize_values=self.normalize_values,
                protect_values_in_text=self.protect_values_in_text,
            ))

        return pd.concat(df_batches_judged)

    def smoke_test(self) -> None:
        """Check that the vLLM server of the judge responds"""
        print(self.llm.root_client.models.list())
        print(self.llm.invoke("What is 17 * 23?").content)

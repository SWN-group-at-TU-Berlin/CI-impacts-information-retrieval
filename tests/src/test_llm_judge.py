import json
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from src import llm_judge as lj


FIELDS = ["infrastructure_type", "location", "damage"]
CHUNK_TEXT = "In Berlin, several roads were flooded. The power supply in Potsdam failed for 3 days."


def make_records():
    return pd.DataFrame({
        "infrastructure_type": ["roads", "power supply", "railway"],
        "location": ["Berlin", "Berlin", "Hamburg"],
        "damage": ["flooded", "failure(s)", "destroyed"],
        "chunk_text": [CHUNK_TEXT] * 3,
    }, index=[10, 11, 12])


def test_records_to_judge_payload():
    df = make_records()
    df.loc[11, "damage"] = np.nan
    payload = json.loads(lj.records_to_judge_payload(df, FIELDS + ["not_a_column"]))
    assert payload[0] == {"record_id": 0, "infrastructure_type": "roads", "location": "Berlin", "damage": "flooded"}
    assert payload[1]["damage"] is None
    assert "chunk_text" not in payload[0]


@pytest.mark.parametrize("response", [
    '[{"record_id": 0, "verdict": "correct"}]',
    'Here is my answer:\n```json\n[{"record_id": 0, "verdict": "correct"}]\n```',
    '<think>record 0 [looks] fine</think>[{"record_id": 0, "verdict": "correct"}]',
    '{"records": [{"record_id": 0, "verdict": "correct"}]}',
])
def test_parse_judge_response(response):
    assert lj.parse_judge_response(response) == [{"record_id": 0, "verdict": "correct"}]


def test_parse_judge_response_invalid():
    with pytest.raises(ValueError):
        lj.parse_judge_response("I cannot judge these records.")


def test_apply_judgements_keeps_all_records():
    judgements = [
        {"record_id": 0, "verdict": "correct", "evidence": "In Berlin, several roads were flooded.", "reason": "stated"},
        {"record_id": 1, "verdict": "adapted", "adapted_fields": {"location": "Potsdam"},
         "evidence": "The power supply in Potsdam failed for 3 days.", "reason": "wrong location"},
        {"record_id": 2, "verdict": "false", "evidence": "", "reason": "railway not mentioned"},
    ]
    result = lj.apply_judgements(make_records(), judgements, FIELDS, chunk_text=CHUNK_TEXT)

    assert len(result) == 3
    assert result.index.tolist() == [10, 11, 12]
    assert result["judge_verdict"].tolist() == ["correct", "adapted", "false"]
    assert result["location"].tolist() == ["Potsdam" if i == 11 else loc for i, loc in zip([10, 11, 12], ["Berlin", "Berlin", "Hamburg"])]
    assert json.loads(result.loc[11, "judge_adapted_fields"]) == {"location": {"old": "Berlin", "new": "Potsdam"}}
    assert result.loc[11, "judge_adaptation_in_text"] == True
    assert result.loc[10, "judge_evidence_in_text"] == True
    # false records are not changed
    assert result.loc[12, ["infrastructure_type", "location", "damage"]].tolist() == ["railway", "Hamburg", "destroyed"]


def test_apply_judgements_missing_and_invalid_judgements():
    judgements = [
        {"record_id": 0, "verdict": "maybe"},          # invalid verdict
        {"record_id": 7, "verdict": "correct"},        # unknown record
        {"record_id": "x", "verdict": "correct"},      # invalid id
    ]
    result = lj.apply_judgements(make_records(), judgements, FIELDS)
    assert result["judge_verdict"].tolist() == [lj.NOT_JUDGED] * 3
    assert result["location"].tolist() == ["Berlin", "Berlin", "Hamburg"]


def test_apply_judgements_ignores_disallowed_adaptations():
    judgements = [
        # chunk_text is not a judged field, damage value not an allowed category -> nothing changes -> correct
        {"record_id": 0, "verdict": "adapted", "adapted_fields": {"chunk_text": "x", "damage": "very wet"}},
        # inferred value not in the text is applied but flagged
        {"record_id": 1, "verdict": "adapted", "adapted_fields": {"location": "Brandenburg", "damage": "outage(s)"}},
    ]
    result = lj.apply_judgements(
        make_records(), judgements, FIELDS, chunk_text=CHUNK_TEXT,
        allowed_values={"damage": ["flooded", "failure(s)", "outage(s)"]},
    )
    assert result.loc[10, "judge_verdict"] == "correct"
    assert result.loc[10, "chunk_text"] == CHUNK_TEXT
    assert result.loc[11, "judge_verdict"] == "adapted"
    assert result.loc[11, "judge_adaptation_in_text"] == False


def test_apply_judgements_set_field_to_null():
    judgements = [{"record_id": 2, "verdict": "adapted", "adapted_fields": {"location": None}}]
    result = lj.apply_judgements(make_records(), judgements, FIELDS)
    assert pd.isna(result.loc[12, "location"])
    assert json.loads(result.loc[12, "judge_adapted_fields"]) == {"location": {"old": "Hamburg", "new": None}}


def test_literal_values():
    annotation = Literal[None, "damaged", "flooded"] | None
    assert lj.literal_values(annotation) == ["damaged", "flooded"]


def test_apply_judgements_does_not_remove_values_stated_in_text():
    judgements = [
        {"record_id": 0, "verdict": "adapted", "adapted_fields": {"location": None, "infrastructure_type": None}},
        {"record_id": 2, "verdict": "adapted", "adapted_fields": {"location": None}},
    ]
    result = lj.apply_judgements(make_records(), judgements, FIELDS, chunk_text=CHUNK_TEXT)

    # "Berlin" and "roads" are in the text -> kept, nothing changed -> correct
    assert result.loc[10, ["infrastructure_type", "location"]].tolist() == ["roads", "Berlin"]
    assert result.loc[10, "judge_verdict"] == "correct"
    # "Hamburg" is not in the text -> can be removed
    assert pd.isna(result.loc[12, "location"])
    assert result.loc[12, "judge_verdict"] == "adapted"

    # without chunk text or with protection switched off, null adaptations are applied
    result = lj.apply_judgements(make_records(), judgements, FIELDS, chunk_text=CHUNK_TEXT, protect_values_in_text=False)
    assert pd.isna(result.loc[10, "location"])


def test_format_judge_report():
    judgements = [
        {"record_id": 0, "verdict": "correct", "reason": "stated"},
        {"record_id": 1, "verdict": "adapted", "adapted_fields": {"location": "Potsdam"}, "reason": "wrong location"},
        {"record_id": 2, "verdict": "false", "reason": "railway not mentioned"},
    ]
    report = lj.format_judge_report(lj.apply_judgements(make_records(), judgements, FIELDS, chunk_text=CHUNK_TEXT))
    lines = report.split("\n")

    assert lines[0] == "LLM-as-a-judge verdicts: {'correct': 1, 'adapted': 1, 'false': 1}"
    assert "  ADAPTED records (1):" in lines
    assert "    record 11: power supply | Berlin | failure(s)" in lines  # values before adaptation
    assert "      location: 'Berlin' -> 'Potsdam'" in lines
    assert "      reason: wrong location" in lines
    assert "    record 12: railway | Hamburg | destroyed" in lines
    assert "record 10" not in report  # correct records are only counted


def test_format_judge_report_single_verdict_string():
    judgements = [
        {"record_id": 1, "verdict": "adapted", "adapted_fields": {"location": "Potsdam"}, "reason": "wrong location"},
        {"record_id": 2, "verdict": "false", "reason": "railway not mentioned"},
    ]
    report = lj.format_judge_report(lj.apply_judgements(make_records(), judgements, FIELDS), verdicts="adapted")
    assert "      location: 'Berlin' -> 'Potsdam'" in report
    assert "FALSE" not in report


def test_apply_judgements_allows_combinations_of_allowed_values():
    df = pd.DataFrame({"economic_impact_origin": ["repair_recovery_cost", "repair_recovery_cost"]})
    allowed = {"economic_impact_origin": ["repair_recovery_cost", "downstream_economic_consequence"]}
    judgements = [
        {"record_id": 0, "verdict": "adapted", "adapted_fields": {"economic_impact_origin": "repair_recovery_cost, downstream_economic_consequence"}},
        {"record_id": 1, "verdict": "adapted", "adapted_fields": {"economic_impact_origin": "repair costs"}},  # not a keyword
    ]
    result = lj.apply_judgements(df, judgements, ["economic_impact_origin"], allowed_values=allowed)
    assert result["economic_impact_origin"].tolist() == ["repair_recovery_cost, downstream_economic_consequence", "repair_recovery_cost"]
    assert result["judge_verdict"].tolist() == ["adapted", "correct"]


@pytest.mark.parametrize("value, expected", [
    ("Two bridges", "bridges"),
    ("the A-7 motorway", "a-7 motorway"),
    ("several roads", "roads"),
    ("1,000 households", "households"),
    ("power supply", "power supply"),
    ("A1 motorway", "a1 motorway"),  # "A1" is a name, not an article
])
def test_strip_determiners(value, expected):
    assert lj.strip_determiners(value) == expected


def test_apply_judgements_ignores_changes_of_only_determiners():
    judgements = [
        {"record_id": 0, "verdict": "adapted", "adapted_fields": {"infrastructure_type": "several roads"}},
        {"record_id": 1, "verdict": "adapted", "adapted_fields": {"infrastructure_type": "the gas network"}},
    ]
    result = lj.apply_judgements(
        make_records(), judgements, FIELDS, normalize_values={"infrastructure_type": lj.strip_determiners},
    )
    assert result.loc[10, "infrastructure_type"] == "roads"
    assert result.loc[10, "judge_verdict"] == "correct"
    # a real change is still applied
    assert result.loc[11, "infrastructure_type"] == "the gas network"
    assert result.loc[11, "judge_verdict"] == "adapted"

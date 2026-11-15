from types import SimpleNamespace
from typing import Literal

import pandas as pd
import pytest
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.models import llm_extract as le
from src.models import llm_judge as lj


class FakeLLM:
    """Chat model returning the given contents in turn"""

    def __init__(self, contents=()):
        self.contents = list(contents)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        content = self.contents.pop(0)
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(content=content, response_metadata={"finish_reason": "stop"})

    def bind_tools(self, tools, **kwargs):
        return SimpleNamespace(tools=tools, kwargs=kwargs)


class FakeGraph:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return {**state, **result}


def make_extraction(results):
    extraction = le.LlmExtract("m", "url", "key", tools=["ner_tool"], llm=FakeLLM())
    extraction.graph = FakeGraph(results)
    return extraction


def test_ci_extraction_binds_tools():
    extraction = le.LlmExtract("m", "url", "key", tools=["ner_tool"], llm=FakeLLM())
    assert extraction.llm_tool.tools == ["ner_tool"]
    assert extraction.llm_tool.kwargs == {
        "tool_choice": "any",
        "parallel_tool_calls": False,
    }


def test_ci_extraction_extract_retries_other_errors():
    extraction = make_extraction([ConnectionError("server down"), {"is_infrastructure_case": True}])
    result = extraction.extract("text", max_retries=1, retry_wait=0)
    assert result == {"text": "text", "is_infrastructure_case": True}
    assert extraction.graph.calls == 2


def test_ci_extraction_extract_does_not_retry_length_limit():
    extraction = make_extraction([le.LengthLimitError("max_tokens"), {}])
    with pytest.raises(le.LengthLimitError):
        extraction.extract("text", max_retries=1, retry_wait=0)
    assert extraction.graph.calls == 1


def test_ci_extraction_extract_requires_compiled_graph():
    with pytest.raises(AssertionError):
        le.LlmExtract("m", "url", "key", llm=FakeLLM()).extract("text")


class Case(BaseModel):
    ci_asset: str = Field(description="Damaged infrastructure asset.")
    damage: Literal[None, "inundation", "disruption"] | None = Field(default=None, description="Damage category.")


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "FIELD RULES:\n{field_rules}"),
        ("human", "TEXT:\n{text}\nNER: {ner_entities}\nRECORDS:\n{records}"),
    ]
)


def make_judge(contents, **kwargs):
    return lj.LlmJudge(
        "m",
        "url",
        "key",
        prompt=JUDGE_PROMPT,
        field_schema={
            "infrastructure_type": (Case, "ci_asset"),
            "damage": (Case, "damage"),
        },
        extra_rules={"damage": ["Use null if no damage is stated."]},
        llm=FakeLLM(contents),
        retry_wait=0,
        **kwargs,
    )


def test_ci_judge_field_rules_and_allowed_values():
    judge = make_judge([])
    assert judge.allowed_values == {"damage": ["inundation", "disruption"]}
    assert "- Allowed values (or null): inundation, disruption" in judge.field_rules
    assert "- Use null if no damage is stated." in judge.field_rules


def test_ci_judge_judges_in_batches():
    df = pd.DataFrame(
        {
            "infrastructure_type": ["roads", "bridges", "rail"],
            "damage": ["inundation"] * 3,
        }
    )
    contents = [
        '[{"record_id": 0, "verdict": "correct"}, {"record_id": 1, "verdict": "adapted", "adapted_fields": {"damage": "disruption"}}]',
        '[{"record_id": 0, "verdict": "false"}]',
    ]
    judge = make_judge(contents, batch_size=2)
    df_judged = judge.judge(df, "Roads were flooded. Bridges were closed. Rail is fine.")
    assert judge.llm.calls == 2
    assert df_judged["judge_verdict"].tolist() == ["correct", "adapted", "false"]
    assert df_judged["damage"].tolist() == ["inundation", "disruption", "inundation"]


def test_ci_judge_is_skipped_after_consecutive_failures():
    df = pd.DataFrame({"infrastructure_type": ["roads"], "damage": ["inundation"]})
    judge = make_judge([ValueError("down")] * 4, max_consecutive_failures=2, max_retries=0)
    for _ in range(3):
        df_judged = judge.judge(df, "Roads were flooded.")
    assert judge.llm.calls == 2  # third chunk is skipped
    assert judge.is_skipped
    assert df_judged["judge_verdict"].tolist() == [lj.NOT_JUDGED]
    judge.reset()
    assert not judge.is_skipped


def test_build_graph_compiles_with_ner_tool():
    from src.models import graph as lg
    from src.tools.ner import get_infrastructure_entities

    extraction = le.LlmExtract("m", "url", "key", tools=[get_infrastructure_entities], llm=FakeLLM())
    graph = lg.build_graph(extraction)
    assert extraction.graph is graph
    assert {
        "is_infrastructure_case",
        "assets",
        "tools",
        "extract_asset_information",
        "detect_cascading_impacts",
        "economic_impacts",
        "societal_impacts",
        "secondary_infrastructure_impacts",
        "environmental_impacts",
    } <= set(graph.nodes)


def test_infrastructure_case_fields_are_named_as_record_columns():
    from src.models.schemas import JUDGE_FIELD_SCHEMA, InfrastructureCase

    case = InfrastructureCase(
        is_infrastructure_case=True,
        infrastructure_type="roads",
        location="Berlin",
        damage="flooded",
    )
    assert case.damage == "inundation"  # free-text damage mapped to its category
    for column in [
        "infrastructure_type",
        "infrastructure_amount",
        "infrastructure_name",
    ]:
        assert column in InfrastructureCase.model_fields
        assert JUDGE_FIELD_SCHEMA[column] == (InfrastructureCase, column)

#!/usr/bin/env python
"""Smoke test of the gpt-oss vLLM server (LLM 1): the same call types as the workflow in notebooks/extract_ciimpacts.py
(plain answer, forced tool call, structured output via tool calling) plus a forced token-limit overrun,
which must be detectable via finish_reason == "length" (used by structured_llm in extract_ciimpacts.py).
Exits with code 1 if one of the checks fails.
usage (from the project root, server running): uv run python run_scripts/smoke_test_gptoss.py
"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # makes "import src..." work from anywhere

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.settings import settings as s


class SmokeCase(BaseModel):
    ci_asset: str
    damage: str


class SmokeCases(BaseModel):
    cases: list[SmokeCase]


@tool
def get_infrastructure_entities(text: str) -> list[str]:
    """Get the infrastructure entities mentioned in the text."""
    return ["railway track"]


def make_llm(max_tokens: int) -> ChatOpenAI:
    # same settings as llm in extract_ciimpacts.py
    return ChatOpenAI(
        model=s.LLM1_MODEL,
        base_url=s.LLM1_BASE_URL,
        api_key=s.VLLM_API_KEY,
        temperature=0,
        seed=42,
        timeout=300,
        max_tokens=max_tokens,
        reasoning_effort="low",
    )


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract the damaged infrastructure assets and their damage from the TEXT."),
        ("human", "TEXT: {text}"),
    ]
)
messages = prompt.format_messages(text="Flood damage to railway track worth tens of millions of euros")

llm = make_llm(max_tokens=8192)
failed = []


def check(name, fn):
    try:
        print(f"[OK]   {name}: {fn()}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        failed.append(name)


print(f"SMOKE TEST of {s.LLM1_MODEL} at {s.LLM1_BASE_URL}")
check("models", lambda: [m.id for m in llm.root_client.models.list().data])
check("plain answer", lambda: llm.invoke("What is 17 * 23? Answer with the number only.").content)


def tool_call():
    calls = llm.bind_tools([get_infrastructure_entities], tool_choice="any", parallel_tool_calls=False).invoke(messages).tool_calls
    assert calls, "no tool call returned"
    return calls


def structured_output():
    out = llm.with_structured_output(SmokeCases, method="function_calling", include_raw=True).invoke(messages)
    assert out["parsed"] is not None, f"no parsed output, finish_reason={out['raw'].response_metadata.get('finish_reason')}"
    return out["parsed"]


def token_limit():
    # 32 tokens are used up by the reasoning: structured_llm relies on parsed=None + finish_reason="length"
    out = make_llm(max_tokens=32).with_structured_output(SmokeCases, method="function_calling", include_raw=True).invoke(messages)
    finish_reason = out["raw"].response_metadata.get("finish_reason")
    assert out["parsed"] is None and finish_reason == "length", f"parsed={out['parsed']}, finish_reason={finish_reason}"
    return f"finish_reason={finish_reason}, usage={out['raw'].usage_metadata}"


check("tool call", tool_call)
check("structured output", structured_output)
check("token limit detected", token_limit)

if failed:
    print(f"SMOKE TEST FAILED: {failed}")
    sys.exit(1)
print("SMOKE TEST PASSED")

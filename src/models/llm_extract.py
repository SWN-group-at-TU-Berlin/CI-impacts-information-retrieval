#!/usr/bin/env python
"""LLM-extractor: extraction of CI impacts from text chunks with a LangGraph workflow (gpt-oss, served by vLLM, see LlmExtract),
the conversion of the graph results into records (impacts_to_dataframe), and the former HF decoder model (DecoderModelCaching)
"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import re
import time

import pandas as pd

from src.settings import init_logger

logger = init_logger(__name__)


class LengthLimitError(RuntimeError):
    """gpt-oss used up max_tokens (mostly for reasoning) before returning the structured answer"""


class LlmExtract:
    """LLM-extractor (e.g. gpt-oss served by vLLM, see scripts/serve_vllm_gptoss.sbatch) for the extraction of CI impacts.

    Holds the chat model, the model with bound tools (NER tool) and the compiled LangGraph workflow.
    The graph nodes, prompts and output schemas are defined in the workflow (notebooks/extract_ciimpacts.py)
    and use .structured(schema) and .llm_tool of this class.

    usage:
        llm_extract = LlmExtract(s.LLM_EXTRACTOR_MODEL, s.LLM_EXTRACTOR_BASE_URL, s.VLLM_API_KEY, tools=[get_infrastructure_entities])
        ...  # define graph nodes with llm_extract.structured(schema) / llm_extract.llm_tool
        llm_extract.compile(builder)
        result = llm_extract.extract(chunk_text)
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        tools: list | None = None,
        temperature: float = 0,
        seed: int = 42,
        timeout: int = 300,
        max_tokens: int = 8192,  # counts reasoning + answer: prompt + max_tokens must be <= vLLM --max-model-len (16384)
        reasoning_effort: str | None = "low",  # gpt-oss: low | medium | high
        llm=None,
    ):
        if llm is None:
            from langchain_openai import ChatOpenAI

            kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
            llm = ChatOpenAI(
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                seed=seed,
                timeout=timeout,
                max_tokens=max_tokens,
                **kwargs,
            )
        self.llm = llm
        self.tools = list(tools or [])
        # bind tool(s) to the graph node(s) which have to call the NER tool first
        self.llm_tool = self.llm.bind_tools(self.tools, tool_choice="any", parallel_tool_calls=False) if self.tools else None
        self.graph = None

    def structured(self, schema):
        """llm.with_structured_output via tool calling, raising LengthLimitError when max_tokens is reached.
        With method="function_calling" langchain returns None instead of raising openai.LengthFinishReasonError,
        which would make the `while result is None` retries loop on the same (deterministic) overrun.
        """
        from langchain_core.runnables import RunnableLambda

        def _parse(output):
            if output["parsed"] is None and output["raw"].response_metadata.get("finish_reason") == "length":
                usage = output["raw"].usage_metadata or {}
                raise LengthLimitError(
                    f"{schema.__name__}: max_tokens reached ({usage.get('output_tokens')} output tokens) "
                    "before the structured answer. Increase max_tokens or lower reasoning_effort of llm."
                )
            if output["parsing_error"] is not None:
                raise output["parsing_error"]  # keep the behaviour of include_raw=False
            return output["parsed"]

        return self.llm.with_structured_output(schema, method="function_calling", include_raw=True) | RunnableLambda(_parse)

    def compile(self, builder):
        """Compile the LangGraph workflow (StateGraph builder) which is run by .extract()"""
        self.graph = builder.compile()
        return self.graph

    @staticmethod
    def is_length_error(error: Exception) -> bool:
        """True if the model reached max_tokens: retrying is deterministic (temperature=0, seed), so it does not help"""
        import openai

        return isinstance(error, LengthLimitError | openai.LengthFinishReasonError)

    def extract(self, text: str, max_retries: int = 1, retry_wait: int = 60, **state) -> dict:
        """Run the graph on a text chunk and return the final graph state.
        Other errors than the token limit are retried after retry_wait seconds (e.g. server not reachable);
        the last error is raised if all attempts failed. Further entries of the initial graph state can be given as **state.
        """
        assert self.graph is not None, "graph not compiled yet, call .compile(builder) first"
        for attempt in range(max_retries + 1):
            try:
                return self.graph.invoke({"text": text, **state})
            except Exception as e:
                if self.is_length_error(e) or attempt == max_retries:
                    raise
                print(f"! ERROR {e}. Sleeping for {retry_wait} s and then try again")
                time.sleep(retry_wait)

    def smoke_test(
        self,
        text: str = "Flood damage to railway track worth tens of millions of euros",
        schema=None,
    ) -> None:
        """Check that the vLLM server responds, incl. tool calling and structured output"""
        from langchain_core.prompts import ChatPromptTemplate

        print(self.llm.root_client.models.list())
        print(self.llm.invoke("What is 17 * 23?").content)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extract the damaged infrastructure assets from the TEXT using the tool.",
                ),
                ("human", "TEXT: {text}"),
            ]
        )
        messages = prompt.format_messages(text=text)
        if self.llm_tool is not None:
            print(self.llm_tool.invoke(messages).tool_calls)
        if schema is not None:
            print(self.structured(schema).invoke(messages))


##########################  LLM-extractor response -> records (one row per infrastructure case)


def normalize_asset(asset):
    """Normalize asset names for matching."""

    if asset is None or pd.isna(asset):
        return ""

    asset = str(asset).lower().strip()

    # Remove leading articles
    asset = re.sub(r"^(the|a|an)\s+", "", asset)
    # Normalize whitespace
    asset = re.sub(r"\s+", " ", asset)

    return asset


def append_value(current, new_value):
    """
    Append a value to an existing semicolon-separated cell.

    Ignores None / NaN / empty values.
    """
    if new_value is None or pd.isna(new_value):
        return current

    new_value = str(new_value).strip()

    if not new_value:
        return current

    if not current:
        return new_value

    return f"{current}; {new_value}"


def impacts_to_dataframe(data):
    # Get extracted information
    infrastructure_cases = data.get("infrastructure_cases", []) or []
    economic_impacts = data.get("economic_impacts", []) or []
    societal_impacts = data.get("societal_impacts", []) or []
    infrastructure_impacts = data.get("secondary_infrastructure_impacts", []) or []
    environmental_impacts = data.get("environmental_impacts", []) or []

    # Create base dataframe
    df = pd.DataFrame([case.model_dump() for case in infrastructure_cases.cases])

    if df.empty:
        return df

    # Normalize asset names for matching
    df["_asset_normalized"] = df["infrastructure_type"].apply(normalize_asset)

    # Initialize output columns
    df["economic_impact_type"] = ""
    df["economic_impact_origin"] = ""
    df["economic_impact_amount"] = ""
    df["economic_impact_unit"] = ""

    df["societal_impact_origin"] = ""
    df["societal_impact_type"] = ""
    df["societal_impact_amount"] = ""
    df["societal_impact_unit"] = ""

    df["environmental_impact_origin"] = ""
    df["environmental_impact_type"] = ""
    df["environmental_impact_amount"] = ""
    df["environmental_impact_unit"] = ""

    df["secondary_infrastructure_asset"] = ""

    # ECONOMIC IMPACTS
    for impact in economic_impacts:
        source_asset = normalize_asset(impact.source_asset)

        if not source_asset:
            continue

        matches = df["_asset_normalized"] == source_asset

        if not matches.any():
            continue

        # -----------------------------------------------------
        # Extract ALL economic impact fields
        impact_type = impact.impact_type
        impact_origin = impact.impact_origin
        impact_amount = impact.amount
        impact_unit = impact.unit

        # Append each field separately
        for idx in df.index[matches]:
            df.at[idx, "economic_impact_type"] = append_value(df.at[idx, "economic_impact_type"], impact_type)
            df.at[idx, "economic_impact_origin"] = append_value(df.at[idx, "economic_impact_origin"], impact_origin)
            df.at[idx, "economic_impact_amount"] = append_value(df.at[idx, "economic_impact_amount"], impact_amount)
            df.at[idx, "economic_impact_unit"] = append_value(df.at[idx, "economic_impact_unit"], impact_unit)

    # SOCIETAL IMPACTS
    for impact in societal_impacts:
        source_asset = normalize_asset(impact.source_asset)

        if not source_asset:
            continue

        impact_origin = impact.impact_origin
        impact_type = impact.impact_type
        impact_amount = impact.amount
        impact_unit = impact.unit

        matches = df["_asset_normalized"] == source_asset

        for idx in df.index[matches]:
            df.at[idx, "societal_impact_origin"] = append_value(df.at[idx, "societal_impact_origin"], impact_origin)
            df.at[idx, "societal_impact_type"] = append_value(df.at[idx, "societal_impact_type"], impact_type)
            df.at[idx, "societal_impact_amount"] = append_value(df.at[idx, "societal_impact_amount"], impact_amount)
            df.at[idx, "societal_impact_unit"] = append_value(df.at[idx, "societal_impact_unit"], impact_unit)

    # ENVIRONMENTAL IMPACTS
    for impact in environmental_impacts:
        source_asset = normalize_asset(impact.source_asset)

        if not source_asset:
            continue

        matches = df["_asset_normalized"] == source_asset

        for idx in df.index[matches]:
            df.at[idx, "environmental_impact_origin"] = append_value(df.at[idx, "environmental_impact_origin"], impact.impact_origin)
            df.at[idx, "environmental_impact_type"] = append_value(df.at[idx, "environmental_impact_type"], impact.impact_type)
            df.at[idx, "environmental_impact_amount"] = append_value(df.at[idx, "environmental_impact_amount"], impact.amount)
            df.at[idx, "environmental_impact_unit"] = append_value(df.at[idx, "environmental_impact_unit"], impact.unit)

    # SECONDARY INFRASTRUCTURE IMPACTS
    for impact in infrastructure_impacts:
        source_asset = normalize_asset(impact.source_asset)

        if not source_asset:
            continue

        impact_target_asset = impact.target_asset

        matches = df["_asset_normalized"] == source_asset

        for idx in df.index[matches]:
            df.at[idx, "secondary_infrastructure_asset"] = append_value(df.at[idx, "secondary_infrastructure_asset"], impact_target_asset)
    # Remove helper column
    df = df.drop(columns="_asset_normalized")

    return df


def result_to_dataframe(result: dict) -> pd.DataFrame:
    """Records (one row per infrastructure case incl. its cascading impacts) of the final graph state of LlmExtract.extract()"""
    df_resp = impacts_to_dataframe(result).replace("; ", ", ", regex=True)
    df_resp["has_cascading_impacts"] = result.get("has_cascading_impacts", []).has_cascading_impacts
    df_resp["assets_org"] = result.get("assets_org", []).cases  # backup to see if assets/loc/dam where modified in CascaImpactExtraction
    return df_resp

#!/usr/bin/env python
"""LangGraph workflow of LLM 1: nodes, routing and graph structure for the extraction of CI impacts.
The graph is built for a LlmExtract (src/models/llm_extract.py) with build_graph(llm_extract).
"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import ast
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send
from pydantic import BaseModel

from src.models.prompts import *  # noqa: F403  prompts of the graph nodes
from src.models.schemas import *  # noqa: F403  output schemas and GraphState
from src.tools.ner import load_ci_patterns

# LLM 1 used by the graph nodes, set by build_graph()
_llm_extract = None


def structured_llm(schema):
    """LLM 1 with structured output (see LlmExtract.structured)"""
    return _llm_extract.structured(schema)


def llm_tool():
    """LLM 1 with the bound NER tool (see LlmExtract.llm_tool)"""
    return _llm_extract.llm_tool


###########################################################################
# Routing
###########################################################################


def should_continue_is_case(state) -> Literal["assets", END]:
    """Route to decide if paragraph text contains infrastructure damages caused by natural hazard"""

    if state["is_infrastructure_case"] is False:
        return END

    return "assets"


def should_continue(state) -> Literal["tools", "extract_asset_information"]:
    """Route to tool handler, or proceed with info-extraction if already conducted tool called"""
    # Get the last message
    print("state", state)
    last_message = state["messages"][-1]
    # If the last message is a tool call, check if it's a Done tool call
    if last_message.tool_calls:
        return "tools"  # run_tool
    # Otherwise, proceed with next node
    return "extract_asset_information"


###########################################################################
# Nodes
###########################################################################


def extract_asset_information(state: GraphState) -> GraphState:
    # Structured-output model
    llm_structured = structured_llm(InfrastructureCases)

    # The ToolNode puts the tool result into the message history
    tool_message = state["messages"][-1]
    assets_list = ast.literal_eval(tool_message.content)  # str -> list

    # remove non-CI types
    ci_patterns = load_ci_patterns()
    assets_list = pp.keep_ci_types(assets_list, ci_patterns)  # (e.g. "24Rhein")
    print("tool_message.content", assets_list)

    messages = [HumanMessage(content=asset_information_prompt.format(text=state["text"], entities=assets_list))]
    result = llm_structured.invoke(messages)
    print("cld:", result)

    i = 0
    while result is None:
        result = llm_structured.invoke(messages)
        try:
            print(f"cld {i}", len(result), result)
        except (TypeError, AttributeError):
            print(f"cld {i}", result)

        if i == 1:
            response_list = []

            llm_structured = structured_llm(InfrastructureCase)

            for asset in assets_list:
                result = llm_structured.invoke([HumanMessage(content=asset_information_prompt.format(text=state["text"], entities=asset))])
                print(f"cld response for {asset}:", result)
                try:
                    response_list.append(result)
                except AttributeError:
                    pass  # avoid to concate enties without Infrastructure case

            # postprocess
            response_list = [x for x in response_list if x is not None]  # keep only InfrastructureCase entries
            result = InfrastructureCases(cases=response_list)

            print("response_list", result)
            return {"assets_org": result, "infrastructure_cases": result}
        i += 1

    return {
        "assets_org": result,
        "infrastructure_cases": result,
    }


def extract_assets(state):
    messages = state.get("messages", [])

    if not messages:
        messages = [HumanMessage(content=asset_prompt.format(text=state["text"]))]

    result = llm_tool().invoke(messages)

    i = 0
    while not result.tool_calls or len(result.tool_calls) == 0:
        print(f"⚠️ No tool calls were returned by the model.Trying again [{i}]")
        result = llm_tool().invoke(messages)
        if i == 5:
            return None
        i += 1

    print("Tool call:", result.tool_calls)

    print("asset m", messages)
    print("asset ", result)

    # NOTE ToolNode expects a list of messages, thus state["assets"] = result does not work
    return {"messages": [result]}


def determine_infrastructure_case(state):
    llm_structured = structured_llm(InfrastructureCase)

    messages = [HumanMessage(content=is_case_prompt.format(text=state["text"]))]

    result = llm_structured.invoke(messages)
    print("is CI Damage", result.is_infrastructure_case)

    return {"is_infrastructure_case": result.is_infrastructure_case}


def detect_cascade(state) -> GraphState:
    class CascadingImpactDetection(BaseModel):
        has_cascading_impacts: bool

    llm_structured = structured_llm(CascadingImpactDetection)
    result = None
    while not result:
        result = llm_structured.invoke(
            [
                HumanMessage(
                    content=prompt_has_cascade.format(
                        text=state["text"],
                        ci_damages=state["infrastructure_cases"].cases,
                    )
                )
            ]
        )
    print("has_cascading_impacts", result)

    return {"has_cascading_impacts": result}


def extract_economic_impacts(state: GraphState) -> GraphState:
    print("state[infrastructure_cases]", state["ci_damages"])

    class EconomicImpacts(BaseModel):
        impacts: list[EconomicImpact]  # any number: empty if none is mentioned, several per asset possible (matched by source_asset)

    llm_structured = structured_llm(EconomicImpacts)

    result = llm_structured.invoke(
        [
            HumanMessage(
                content=economic_impact_prompt.format(
                    text=state["text"],
                    ci_damages=state["ci_damages"],
                )
            )
        ]
    )

    try:
        print("economic impacts", result.impacts)
        return {"economic_impacts": result.impacts}
    except:
        return {"economic_impacts": None}


def extract_infrastructure_impacts(state: GraphState) -> GraphState:
    class SecondaryInfrastructureImpacts(BaseModel):
        impacts: list[SecondaryInfrastructureImpact]  # any number: empty if none is mentioned, several per asset possible (matched by source_asset)

    llm_structured = structured_llm(SecondaryInfrastructureImpacts)

    result = llm_structured.invoke(
        [
            HumanMessage(
                content=infrastructure_impact_prompt.format(
                    text=state["text"],
                    ci_damages=state["ci_damages"],
                )
            )
        ]
    )

    try:
        print("secondary_infrastructure_impacts", result)
        return {"secondary_infrastructure_impacts": result.impacts}
    except:
        return {"secondary_infrastructure_impacts": None}


def extract_societal_impacts(state: GraphState) -> GraphState:
    class SocietalImpacts(BaseModel):
        impacts: list[SocietalImpact]  # any number: empty if none is mentioned, several per asset possible (matched by source_asset)

    llm_structured = structured_llm(SocietalImpacts)

    result = llm_structured.invoke(
        [
            HumanMessage(
                content=societal_impact_prompt.format(
                    text=state["text"],
                    ci_damages=state["ci_damages"],
                )
            )
        ]
    )

    try:
        print("societal_impacts", result)
        return {"societal_impacts": result.impacts}
    except:
        return {"societal_impacts": None}


def extract_environmental_impacts(state: GraphState) -> GraphState:
    class EnvironmentalImpacts(BaseModel):
        impacts: list[EnvironmentalImpact]  # any number: empty if none is mentioned, several per asset possible (matched by source_asset)

    llm_structured = structured_llm(EnvironmentalImpacts)

    result = llm_structured.invoke(
        [
            HumanMessage(
                content=environmental_impact_prompt.format(
                    text=state["text"],
                    ci_damages=state["ci_damages"],
                )
            )
        ]
    )

    try:
        print("environmental_impacts", result)
        return {"environmental_impacts": result.impacts}
    except:
        return {"environmental_impacts": None}


def route_cascading_impacts(state: GraphState):
    """If cascading impacts exists, extract them"""

    return [
        Send(
            "economic_impacts",
            {
                "text": state["text"],
                "ci_damages": state["infrastructure_cases"].cases,
            },
        ),
        Send(
            "societal_impacts",
            {
                "text": state["text"],
                "ci_damages": state["infrastructure_cases"].cases,
            },
        ),
        Send(
            "secondary_infrastructure_impacts",
            {
                "text": state["text"],
                "ci_damages": state["infrastructure_cases"].cases,
            },
        ),
        Send(
            "environmental_impacts",
            {
                "text": state["text"],
                "ci_damages": state["infrastructure_cases"].cases,
            },
        ),
    ]


###########################################################################
# Graph
###########################################################################


def build_graph(llm_extract):
    """Build and compile the LangGraph workflow of LLM 1 for the given LlmExtract (src/models/llm_extract.py).
    The graph is also stored as llm_extract.graph, run it with llm_extract.extract(text).

    NOTE NIviDiA API LIMIT: dont do parallel specialist calls, or parallel nodes
    "hitting concurrency limits. You could deliberately make the two specialist agents sequential if rate/concurrency limits become a problem."
    """
    global _llm_extract
    _llm_extract = llm_extract

    builder = StateGraph(GraphState)

    builder.add_node("is_infrastructure_case", determine_infrastructure_case)
    builder.add_node("assets", extract_assets)
    builder.add_node("tools", ToolNode(llm_extract.tools))
    builder.add_node("extract_asset_information", extract_asset_information)
    builder.add_node("detect_cascading_impacts", detect_cascade)
    builder.add_node("economic_impacts", extract_economic_impacts)
    builder.add_node("societal_impacts", extract_societal_impacts)
    builder.add_node("secondary_infrastructure_impacts", extract_infrastructure_impacts)
    builder.add_node("environmental_impacts", extract_environmental_impacts)

    # Set arcs
    builder.set_entry_point("is_infrastructure_case")
    builder.add_conditional_edges("is_infrastructure_case", should_continue_is_case, ["assets", END])
    builder.add_conditional_edges("assets", should_continue, ["tools", "extract_asset_information"])
    builder.add_edge("tools", "extract_asset_information")
    # Cascading-impacts
    builder.add_edge("extract_asset_information", "detect_cascading_impacts")
    builder.add_conditional_edges("detect_cascading_impacts", route_cascading_impacts)

    return llm_extract.compile(builder)

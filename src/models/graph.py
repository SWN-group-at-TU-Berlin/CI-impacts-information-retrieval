#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LangGraph workflow of LLM 1: nodes, routing and graph structure for the extraction of CI impacts.
The graph is built for a LlmExtract (src/models/llm_extract.py) with build_graph(llm_extract)."""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import ast
import re
import time
from typing import List, Literal

import numpy as np
import pandas as pd
from langchain.agents.middleware import wrap_tool_call, ModelRequest
from langchain.messages import ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from src.models.schemas import *  # noqa: F401,F403  output schemas and GraphState
from src.models.prompts import *  # noqa: F401,F403  prompts of the graph nodes
from src.tools.ner import get_infrastructure_entities


# LLM 1 used by the graph nodes, set by build_graph()
_llm_extract = None


def structured_llm(schema):
    """LLM 1 with structured output (see LlmExtract.structured)"""
    return _llm_extract.structured(schema)


def llm_tool():
    """LLM 1 with the bound NER tool (see LlmExtract.llm_tool)"""
    return _llm_extract.llm_tool


# IMPORTANT:
# - Do NOT verify the locations.
# - Do NOT correct the locations.
# - Do NOT reinterpret the locations.
# - Do NOT replace a location with another location.
# - Do NOT infer a different location.
# - Treat the extracted locations as authoritative.
# - Pass the TEXT information to "get_coordinates_and_meta_info" tool as input.
# - Return the geographic information provided by GeoLLM.

# FIELDS TO EXTRACT:
# The geographic information should include, where available:
# - location name
# - latitude
# - longitude
# - OSM ID
# - OSM type
# - OSM key
# - OSM value
# - label
# - city
# - state
# - country
# - geometry
# """),


# NOTE
# returns often Workers overload
# Exception: [503] {'message': 'ResourceExhausted: Worker local total request limit reached (16/16)', 'type': 'Service Unavailable', 'code': 503}
# {'error': {'message': 'ResourceExhausted: Worker local total request limit reached (16/16)', 'type': 'Service Unavailable', 'code': 503}, 'status': 503}
def enrich_locations(state):
    # NOTE not part of the graph at the moment, llm_enrichment_tools (GeoLLM tool) is not defined


    # class EnrichedInfrastructureCases(BaseModel):
    #     enriched_cases: list[EnrichedInfrastructureCase]

    # llm_structured = llm_enrichment.with_structured_output(EnrichedInfrastructureCases)


    # call tool, when not done previously
    geo_messages = state["messages"]
    if len(geo_messages) == 0:
        
        # extract loc coord for each single loc previouls identified:
        loc_list = []
        for locs in state["locations"]: 

            if locs == np.nan or locs == "nan" or locs == "NAN" or locs == "NaN" or locs == "":
                loc_list.append("NAN")
            
            locs = re.split(r" and | or | to ", locs)

            for loc in locs:
                geo_messages = [
                    HumanMessage(
                    content=enrich_locations_prompt.format(
                        text=state["text"],
                        locations=loc,
                ))]
                result = llm_enrichment_tools.invoke(geo_messages)
                print("loc geo_messages", loc)
                print("loc geo_messages", geo_messages)
                print("loc geotool", result)

                # i = 0    
                # while not result.tool_calls or len(result.tool_calls) == 0:
                #     print(f"⚠️ No Geo-tool call was returned by the model. Trying again [{i}]")
                #     result = llm_enrichment_tools.invoke(geo_messages)        
                #     if i == 5: 
                #         return None
                #     i += 1
                # else:
                #     print("! tool called:", result.tool_calls)
                        
                # print("enrich_locations m", geo_messages)
                # print("enrich_locations ", result)

                try:
                    loc_list.append(result)
                except AttributeError:
                        loc_list.append("NAN")
                return {"messages": loc_list}


    # try:
    #     result = state["messages"][-1].content #.split("[")
    #     print("LOCS finalize", len(ast.literal_eval(result)), result)

    #     result = ast.literal_eval(result) # make str[list[str]] -> list[str]
    #     return {"enriched_cases": result}
    # except Exception as e:
    #     print("e", e)
    #     # NOTE ToolNode expects a list of messages, thus state["assets"] = result does not work
    #     return {"messages": [result]}


# def enrichment_finalize(state: GraphState) -> GraphState:

#     llm_structured = llm_enrichment.with_structured_output(
#         EnrichedInfrastructureCases
#     )

#     try:
#         result = llm_structured.invoke(
#             [
#    ("system",
# """
# Combine the extracted information for each case. 
# Take the first element from each field (ASSETS, LOCATIONS, DAMAGES) and combine them into a dictionary, proceed with the second element of each field and combine them into a second dictionary, and so on..

# Return one or more dictionaries.

# """),
#     ("human",
# """
# ASSETS: {assets}

# LOCATIONS: {locations}

# DAMAGES: {damages}

# """)

#             HumanMessage(
#                 content=f"""
# Create the final infrastructure cases by combining the extracted
# information with the geographic metadata returned by GeoLLM.

# # Assets:
# # {state["assets"]}

# # Locations:
# # {state["locations"]}

# # Damages:
# # {state["damages"]}

# # GeoLLM result:
# # {state["messages"][-1].content}

# Rules:

# 1. Keep the asset exactly as extracted.
# 2. Keep the location exactly as extracted.
# 3. Keep the damage exactly as extracted.
# 4. Only add geographic information returned by GeoLLM.
# 5. Do not verify or reinterpret the location.
# 6. Do not invent coordinates or OSM metadata.
# 7. If GeoLLM did not return information, use null.
# 8. Preserve the GeoLLM geometry exactly.
# """
#     )])
#     except Exception  as e: 
#         print(f"! ERROR {e}. Sleeping for 1 min and then try again")
#         time.sleep(60)
#         result = llm_structured.invoke(
#             [
#             HumanMessage(
#                 content=f"""
# Create the final infrastructure cases by combining the extracted
# information with the geographic metadata returned by GeoLLM.

# Assets:
# {state["assets"]}

# Locations:
# {state["locations"]}

# Damages:
# {state["damages"]}

# GeoLLM result:
# {state["messages"][-1].content}

# Rules:

# 1. Keep the asset exactly as extracted.
# 2. Keep the location exactly as extracted.
# 3. Keep the damage exactly as extracted.
# 4. Only add geographic information returned by GeoLLM.
# 5. Do not verify or reinterpret the location.
# 6. Do not invent coordinates or OSM metadata.
# 7. If GeoLLM did not return information, use null.
# 8. Preserve the GeoLLM geometry exactly.
# """
#     )])
#     print(result)
#     return {"enriched_cases": result.cases}


# #### Tools - NER + GEoLLM


###########################################
# Tools
###########################################



###########################################
# Tools
###########################################


## NOTE: info for tool calls:
# https://medium.com/@vivekvjnk/introduction-to-tool-use-with-langgraphs-toolnode-0121f3c8c323
# https://docs.langchain.com/oss/python/langchain/tools


## NOTE
# ## code snippets: https://github.com/rabikumarsingh/Create-an-agent-and-integrate-Tavily-search-as-a-tool/blob/main/emailAgent.ipynb
# def run_tool(state: MessagesState):
#     """Performs the tool call"""
#     result = []
#     for tool_call in state["messages"][-1].tool_calls:
#         observation = get_infrastructure_entities.invoke(tool_call["args"])
#         result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
#     return {"messages": result}


def should_continue_is_case(state) -> Literal["assets", END]:
    """Route to decide if paragraph text contains infrastructure damages caused by natural hazard"""

    if state["is_infrastructure_case"] == False:
        return END

    return "assets"


def should_continue(state) -> Literal["tools", "extract_asset_information"]:
    """Route to tool handler, or proceed with info-extraction if already conducted tool called"""
    # Get the last message
    print("state", state)
    last_message = state["messages"][-1]
    # If the last message is a tool call, check if it's a Done tool call
    if last_message.tool_calls:
        return "tools" # run_tool
    # Otherwise, proceed with next node
    return "extract_asset_information"


# get error messages when tool is wrongly used
# @wrap_tool_call
# def handle_tool_errors(
#     request: ToolCallRequest,
#     handler: Callable[[ToolCallRequest], ToolMessage],
# ) -> ToolMessage:
#     """Convert tool exceptions into ToolMessages the model can handle."""
#     try:
#         return handler(request)
#     except Exception as e:
#         return ToolMessage(
#             content=f"Tool error: Please check your input and try again. ({e})",
#             tool_call_id=request.tool_call["id"],
#         )

@wrap_tool_call
def handle_tool_errors(request : ModelRequest, handler) -> ToolMessage:
   print("#### AGENT ABOUT TO CALL TOOL (wrap-style hook) #####")
   print(f"=> TOOL: {request.tool_call['name']}")
   try:
       return handler(request)
   except Exception as e:
       return ToolMessage(
           content=f"TOOL ERROR: Please check your input and try again. ({str(e)})",
           tool_call_id=request.tool_call["id"]
       )

# # bind tool two 2nd model
# geo_tools = [get_coordinates_and_meta_info]
# geo_tool_node = ToolNode(geo_tools)
# llm_enrichment_tools = llm_enrichment.bind_tools(
#     geo_tools,
#     tool_choice="any",
#     parallel_tool_calls=False,
# )


## create agent


# ## create agent
# llm_tool = create_agent(
#     model=llm_tool, 
#     # system_prompt="You are a helpful assistant",
#     middleware=[handle_tool_errors],
# )


# ##### Graph components

###########################################################################
# Node 1
###########################################################################
from langchain_core.messages import HumanMessage


def extract_asset_information(state: GraphState) -> GraphState:

    # Structured-output model
    llm_structured = structured_llm(InfrastructureCases)

    # The ToolNode puts the tool result into the message history
    tool_message = state["messages"][-1]
    assets_list = ast.literal_eval(tool_message.content)  # str -> list

    # remove non-CI types
    ci_patterns = pd.read_json("./ner_patterns.jsonl/patterns", lines=True)
    assets_list = pp.keep_ci_types(assets_list, ci_patterns) # (e.g. "24Rhein")
    print("tool_message.content", assets_list)


    messages = [HumanMessage(
        content=asset_information_prompt.format(
            text=state["text"],
            entities=assets_list
    ))]
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
                result = llm_structured.invoke(
                    [HumanMessage(
                        content=asset_information_prompt.format(
                            text=state["text"],
                            entities=asset
                    ))]
                )
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

    return {"assets_org": result, "infrastructure_cases": result,}

     
def extract_assets(state):

    messages = state.get("messages", [])

    if not messages:
        messages = [
            HumanMessage(
                content=asset_prompt.format(
                    text=state["text"]
                )
            )
        ]

    result = llm_tool().invoke(messages)

    # if not result.tool_calls:
    #     raise RuntimeError(
    #         "LLM did not call get_infrastructure_entities."
    #     )
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
    
    # # if response is not empty, proceed with location extraction
    # if result.content:
    # state["assets"] = result #state["messages"][-1].content # result
    # return state
    # else:        
    # NOTE ToolNode expects a list of messages, thus state["assets"] = result does not work
    return {"messages": [result]}
   

def determine_infrastructure_case(state):

    llm_structured = structured_llm(InfrastructureCase)
    
    messages = [
        HumanMessage(
            content=is_case_prompt.format(
                text=state["text"]
    ))]

    result = llm_structured.invoke(messages)
    print("is CI Damage", result.is_infrastructure_case)
    
    return {"is_infrastructure_case": result.is_infrastructure_case}
   

##########################################################################
# Conditional Node
###########################################################################

# process tool call into 
def asset_finalize(state):

    # result = llm_structured.invoke([
    #     HumanMessage(
    #         content=asset_finalize_prompt.format(
    #         message= state["messages"]
    #         )
    # )])
    # make as List[str]
    result = state["messages"][-1].content #.split("[")
    # result = result[1].split("]")[0].split(",")#
    print("Asset finalize", len(ast.literal_eval(result)), result)

    # if result in ["None", "null", None]:

    # if not result in ["None", "null", None]:
    result = ast.literal_eval(result) # make str[list[str]] -> list[str]

    # remove non-CI types
    ci_patterns = pd.read_json("./ner_patterns.jsonl/patterns", lines=True)
    result = pp.keep_ci_types(result, ci_patterns) # (e.g. "24Rhein")
    

    return {"assets": result}

    # state["assets"] = result
    # return state


###########################################################################
# Node 2
###########################################################################
from typing import Annotated
from annotated_types import Len

def extract_locations(state) -> GraphState:

    class Location(BaseModel):
        location: Annotated[List[str], Len(min_length=len(state["assets"]), max_length=len(state["assets"]))]
                # Dict[str, List[str]] #List[str]
        # longitude: List[float]
        # latitude: List[float]
        # rag_estimated: List[bool]

    # bind PydanticSchema to LLM
    llm_structured = structured_llm(Location)

    result = llm_structured.invoke([
                HumanMessage(
                    content=location_prompt.format(
                        text=state["text"],
                        assets=state["assets"],
                ))
            ])
    print("locs", result)


    # try again if could not extract location (None) or location info is shorter than no. of assets
    # TODO force LLM wiht LangGraph func/method to output response in certain length
    i = 0
    while (result is None) or (len(result.location) < len(state["assets"])):

        result = llm_structured.invoke([
                HumanMessage(
                    content=location_prompt.format(
                        text=state["text"],
                        assets=state["assets"]
        ))])
        try:
            print(f"locs {i}", len(result.location), result)
        except (TypeError, AttributeError):
            print(f"locs {i}", result)

        # extract locations for each single asset, when extraction takes too long
        if i == 5:
            assets_list = state["assets"]
            location_list = []

            class SingleLocation(BaseModel):
                location: Annotated[List[str], Len(min_length=1, max_length=1)] | str="NAN"
            # bind PydanticSchema to LLM
            llm_structured = structured_llm(SingleLocation)

            for asset in assets_list:
                result = llm_structured.invoke([
                        HumanMessage(
                            content=location_prompt.format(
                                text=state["text"],
                                assets=asset
                ))])
                try:
                    location_list.append(result.location)
                except AttributeError:
                    location_list.append("NAN")
            print("locations_list", location_list)
            return {"locations": location_list}
        i += 1

    # else:
    return {"locations": result.location}

###########################################################################
# Node 3
###########################################################################

def extract_damage(state) -> GraphState:

    # define Output Schema and bind it to LLM
    class Damage(BaseModel):
        damage: Annotated[List[str], Len(min_length=len(state["assets"]), max_length=len(state["assets"]))]

    llm_structured = structured_llm(Damage)

    result = llm_structured.invoke(
        [HumanMessage(
            content=damage_prompt.format(
                text=state["text"],
                assets=state["assets"],
                # locations=state["locations"]
    ))])


    print("damage m:", HumanMessage(
                content=damage_prompt.format(
                    text=state["text"],
                    assets=state["assets"],
                    # assets_length=len(state["assets"]),
                    # locations=state["locations"]
    )))
    print("damages", result)

    i = 0
    while (result is None) or (len(result.damage) < len(state["assets"])):
    # while (result is None) or (len(result.location) < len(ast.literal_eval(state["assets"]))):
        result = llm_structured.invoke([
                HumanMessage(
                    content=damage_prompt.format(
                        text=state["text"],
                        assets=state["assets"]
        ))])
        try:
            print(f"damages {i}", len(result.damage), result)
        except (TypeError, AttributeError):
            print(f"damages {i}", result)

        # extract locations for each single asset, when extraction takes too long
        if i == 5:

            class SingleDamage(BaseModel):
                damage: Annotated[List[str], Len(min_length=1, max_length=1)] | str="NAN"
            # bind PydanticSchema to LLM
            llm_structured = structured_llm(SingleDamage)

            assets_list = state["assets"]
            damages_list = []
            for asset in assets_list:
                result = llm_structured.invoke([
                        HumanMessage(
                            content=damage_prompt.format(
                                text=state["text"],
                                assets=asset
                ))])
                try:
                    # print(type(result.damage), result.damage)
                    damages_list.append(result.damage)
                except AttributeError:
                    damages_list.append("NAN")
            print("damagess_list", damages_list)
            # return {"damages": str(damages_list)}
            return {"damages": damages_list}

        i += 1


    return {
        "damages": result.damage
    }

###########################################################################
# Node 4
###########################################################################


###########################################################################
# Node 5
###########################################################################
 

def verify(state):

    # define Output Schema
    llm_structured = structured_llm(InfrastructureCases)
    
    chain = verification_prompt | llm_structured

    result = chain.invoke({
        "text": state["text"],
        "cases": state["extracted_cases"]
    })

    print("verify", result)

    return {
        "secondary_effects": result.secondary_effects,
        "affected_asset_or_services": result.affected_asset_or_services
    }


# Determine for every damaged infrastructure asset or service (ASSETS) whether it causes a downstream impact according to the TEXT.

# For every infrastructure asset or service returned by ASSETS determine whether the following TEXT mentions also one or multiple cascading impacts caused by the failure or disruption of this infrastructure, i.e. determine for each infrastructure failure if it led to one or multiple downstream impacts.
# A cascading impact occurs when: 
# An a single infrastructure asset or service (ASSETS) is damaged or disrupted, AND causes a further impact, such as economic costs, affects people or households, or affects another infrastructure asset or service.


#     Determine for each infrastructure failure (DAMAGED_ASSET) if the TEXT mentioned that it led to one or multiple downstream impacts. 
# A downstream impact occurs when: A single infrastructure asset or service is damaged or disrupted (DAMAGED_ASSET), AND caused a further impact, e.g., caused economic costs, affects people or households, or affects another infrastructure asset or service.

# ITEM TO EXTRACT:
# - "has_cascading_impacts": "List of booleans indicating for each affected infrastructure (DAMAGED_ASSET) if it caused downstream impacts."

# Example:
# infrastructure_type:["roads","railway infrastructure"]
# damage: ["blocked", "flooded"]
# has_cascading_impacts:[True, False]


# class InfrastructureCase(BaseModel):
#     asset: str
#     location: str
#     damage: str

# class InfrastructureCases(BaseModel):
#     impact_cases: list[InfrastructureCase]


def detect_cascade(state) -> GraphState:

    # # merge cases first
    # assets = state["assets"]
    # damages = state["damages"]
    # locations = state["locations"]

    # ci_cases = InfrastructureCases(
    #     impact_cases = [
    #         InfrastructureCase(asset=a, location=l, damage=d) for a, l, d in zip(assets, locations, damages)
    # ])

    class CascadingImpactDetection(BaseModel):
        has_cascading_impacts: bool
        # has_cascading_impacts: Annotated[List[bool], Len(min_length=len(state["assets"]), max_length=len(state["assets"]))]

    llm_structured = structured_llm(CascadingImpactDetection)
    result = None
    while not result:
        result = llm_structured.invoke([
            HumanMessage(content=prompt_has_cascade.format(
                text=state["text"],
                ci_damages=state["infrastructure_cases"].cases,
        ))])
    print("has_cascading_impacts", result)

    return {"has_cascading_impacts": result}


def extract_economic_impacts(state: GraphState) -> GraphState:

    print("state[infrastructure_cases]", state["ci_damages"])
    class EconomicImpacts(BaseModel):
        # impacts: list[EconomicImpact]
        impacts: List[EconomicImpact]  # any number: empty if none is mentioned, several per asset possible (matched by source_asset)

    llm_structured = structured_llm(EconomicImpacts)

    result = llm_structured.invoke([
        HumanMessage(content=economic_impact_prompt.format(
            text=state["text"],
            ci_damages=state["ci_damages"],
       ))])

    try:
        print("economic impacts", result.impacts)
        return {"economic_impacts": result.impacts}
    except:
        return {"economic_impacts": None}


# IMPORTANT:

# 1. Extract ONLY downstream infrastructure or infrastructure-service impacts.
# 2. Do NOT report the original infrastructure damage itself.
# 3. The downstream impact must be explicitly stated or clearly described
#    in the TEXT.
# 4. Do NOT infer impacts that are not supported by the TEXT.
# 5. Do NOT invent affected infrastructure, services, or consequences.
# 6. Do NOT extract economic impacts such as repair costs or financial losses.
# 7. Do NOT extract societal impacts such as affected households or people.
# 8. If no infrastructure/service cascading impact is explicitly mentioned, return an empty list.

# For each impact, identify:
# - the original/source infrastructure asset
# - the original damage
# - the affected infrastructure asset or service
# - the downstream impact


def extract_infrastructure_impacts(state: GraphState) -> GraphState:

    class SecondaryInfrastructureImpacts(BaseModel):
        # impacts: list[SecondaryInfrastructureImpact]
        impacts: List[SecondaryInfrastructureImpact]  # any number: empty if none is mentioned, several per asset possible (matched by source_asset)

    llm_structured = structured_llm(SecondaryInfrastructureImpacts)

    result = llm_structured.invoke([
            HumanMessage(content=infrastructure_impact_prompt.format(
                text=state["text"],
                ci_damages=state["ci_damages"],
        ))])

    try:
        print("secondary_infrastructure_impacts", result )
        return {"secondary_infrastructure_impacts": result.impacts }
    except:
        return {"secondary_infrastructure_impacts": None}


# IMPORTANT:

# 1. Extract ONLY societal or population-level consequences.
# 2. Do NOT report the original infrastructure damage itself.
# 3. Do NOT extract monetary costs or economic losses.
# 4. Do NOT extract infrastructure-to-infrastructure impacts such as
#    train delays or electricity outages unless they explicitly result
#    in a societal consequence.
# 5. The societal impact must be explicitly stated in the TEXT.
# 6. Do NOT infer population numbers.
# 7. Do NOT invent affected households, people, communities, or other
#    societal impacts.
# 8. Preserve quantitative information exactly as stated in the text.
# 9. If no societal impact is explicitly mentioned, return an empty list.

# For each impact identify:

# - the infrastructure asset causing the impact
# - the original damage
# - the affected population
# - the societal consequence
# - the quantitative magnitude, if explicitly provided


def extract_societal_impacts(state: GraphState) -> GraphState:

    class SocietalImpacts(BaseModel):
        # impacts: list[SocietalImpact]
        impacts: List[SocietalImpact]  # any number: empty if none is mentioned, several per asset possible (matched by source_asset)

    llm_structured = structured_llm(SocietalImpacts)


    result = llm_structured.invoke([
            HumanMessage(content=societal_impact_prompt.format(
                text=state["text"],
                ci_damages=state["ci_damages"],
        ))])
    
    try:
        print("societal_impacts", result )
        return {"societal_impacts": result.impacts }
    except:
        return {"societal_impacts": None}


def extract_environmental_impacts(state: GraphState) -> GraphState:

    class EnvironmentalImpacts(BaseModel):
        impacts: List[EnvironmentalImpact]  # any number: empty if none is mentioned, several per asset possible (matched by source_asset)

    llm_structured = structured_llm(EnvironmentalImpacts)

    result = llm_structured.invoke([
            HumanMessage(content=environmental_impact_prompt.format(
                text=state["text"],
                ci_damages=state["ci_damages"],
        ))])

    try:
        print("environmental_impacts", result )
        return {"environmental_impacts": result.impacts }
    except:
        return {"environmental_impacts": None}


from langgraph.types import Send

def route_cascading_impacts(state: GraphState):
    """ If cascading impacts exists, extract them"""
    
    # if not state["has_cascading_impacts"].has_cascading_impacts:
    #     return END
    # print("state", state)
    return [
        Send("economic_impacts", {
            "text": state["text"],
            "ci_damages": state["infrastructure_cases"].cases,
        }),
        Send(
            "societal_impacts",
            {
                "text": state["text"],
                "ci_damages": state["infrastructure_cases"].cases,
        }),
        Send(
            "secondary_infrastructure_impacts",
            {
                "text": state["text"],
                "ci_damages": state["infrastructure_cases"].cases,
            }
        ),
        Send(
            "environmental_impacts",
            {
                "text": state["text"],
                "ci_damages": state["infrastructure_cases"].cases,
            }
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

    ## add memory
    # memory = MemorySaver()
    # graph = builder.compile(checkpointer=memory)

    return llm_extract.compile(builder)

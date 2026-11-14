#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Output schemas (pydantic) of LLM 1 (CI impact extraction), the state of the LangGraph workflow, and the mapping of
the record columns to these schemas for the LLM-as-a-judge (JUDGE_FIELD_SCHEMA)"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

from typing import List, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel

import src.postprocess as pp




# # text = "A few national highways were partly flooded (e.g. the A76 in both directions) or briefly closed (&lt; 3 d) because of the potential of flooding. Most likely due to relative low-flow velocities, damage to Dutch national road infrastructure was limited. Several railway sections were closed (e.g. the railway section between Maastricht and Liége) and some damage occurred to the railway infrastructure, in particular to the electronic 'track circuit' devices and saturated railway embankments (Prorail, 2021).   "
# text = "Similar to Germany, severe damage had been observed to the gas network. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021). In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021)."
# # print(text)
# get_infrastructure_location_pairs(text)#[["ci_entity", "geo_entity","geo_2_entity","token_distance", "token_2_distance"]]

# #### LangGraph OutputSchema and Graph

from shapely.geometry import Point, Polygon, MultiPolygon, MultiLineString, LineString
from pydantic_shapely import GeometryField
from pydantic import Field, PrivateAttr, field_validator
import typing


def coerce_category(value, keywords: dict, fallback: str):
    """Map a free-text category (tool calling does not enforce Literal values, e.g. 'insured_loss') to an allowed one:
    allowed values and None are kept, else the first category whose keyword occurs in the value, else the fallback."""
    if value is None or value in keywords or value == fallback:
        return value
    text = str(value).lower()
    return next((category for category, words in keywords.items() if any(w in text for w in words)), fallback)


ECONOMIC_ORIGIN_KEYWORDS = {
    "repair_recovery_cost": ["direct", "repair", "recover", "restor", "reconstruct", "rebuild", "replac"],
    "downstream_economic_consequence": ["downstream", "indirect", "business", "employe", "workers", "customer", "companies", "company", "interrupt", "supply", "production", "revenue", "trade", "freight", "logistic"],
}
SOCIETAL_ORIGIN_KEYWORDS = {
    "evacuation_or_displacement": ["evacuat", "displac", "relocat", "homeless", "rescu"],
    "loss_of_critical_service_or_inaccessibility": ["service", "access", "without", "isolat", "cut off", "outage", "supply"],
    "disruption_of_mobility": ["mobility", "travel", "commut", "passenger", "strand", "journey"],
}
# only human-related impact types, specific groups first, generic 'people' last
# NOTE: 'residential_area' before 'people', otherwise "residential" would match "resident"
SOCIETAL_TYPE_KEYWORDS = {
    "households": ["household", "homes", "famil"],
    "students": ["student", "pupil", "school"],
    "passengers": ["passenger", "commut", "traveller", "traveler"],
    "community": ["region", "communit", "municipalit", "village", "residential", "neighbourhood", "neighborhood"],
    "population": ["population", "inhabitant", "citizen"],
    "people": ["people", "person", "children", "elderl", "resident", "individual", "patient", "cases", "victim", "evacu", "fatalit", "injur"],
}


class EconomicImpact(BaseModel):
    source_asset: str = Field(
        description=("Infrastructure asset or service whose damage or disruption caused the economic impact."
    ))
    source_location: str | None = Field(
        default=None,
        description=("Location of the infrastructure asset associated with the economic impact."
        ))
    impact_origin: Literal[None, "repair_recovery_cost","downstream_economic_consequence", "other_costs_or_losses"] = Field(
        default=None,
        description=("Whether the impact is a direct cost of repairing, restoring, replacing or recovering damaged infrastructure, or a downstream economic consequence caused by infrastructure damage or disruption."
    ))
    impact_type: str | None = Field(
        default=None,
        description=("Specific type of economic impact, for example: affected employees, affected companies, affected customers, repair cost, reconstruction/replacement cost, total/general cost, total/general losses, business interruption, supply-chain disruption, production interruption or loss, freight or logistics or trade disruption, revenue loss, agricultural loss, insurance loss, additional operating cost, or other explicitly reported economic consequence."
    ))
    # damage_type: str | None = Field(
    #     default=None,
    #     description=("Original damage, failure or disruption affecting the source infrastructure."
    # ))
    # description: str = Field(
    #     description=("Concise description of the economic impact explicitly stated in the source text."
    # ))
    amount: str | None = Field(
        default=None,
        description=("Quantitative costs explicitly reported in the text, if any. Preserve the original wording and do not calculate or infer values. Examples: '€5 million', 'more than £20,000', 'approximately 10 000 Euros'."
    ))
    unit: str | None = Field(
        default=None,
        description=("Currency or unit associated with amount, if explicitly stated. Examples: 'EUR', '€', 'euros', 'GBP'"
    ))

    # tool calling does not enforce the Literal: e.g. 'direct_cost' -> 'repair_recovery_cost', 'insured_loss' -> 'other_costs_or_losses'
    _map_impact_origin = field_validator("impact_origin", mode="before")(
        lambda v: coerce_category(v, ECONOMIC_ORIGIN_KEYWORDS, fallback="other_costs_or_losses"))


# class EconomicImpact(BaseModel):
#     # description: str
#     source_asset: str | None = Field(default=None, description="Infrastructure asset or service whose damage or disruption caused the economic impact.")
#     source_location: str | None = Field(default=None, description="Original location of the source asset.")
#     impact_type: str | None = Field(default=None, description="The type of the economic impact caused by the damaged infrastructure (qualitative information).")
#     # e.g. impact-type: freight industry, industrial disruptions, supply-chains, agricultural
#     # damage_type: str | None = Field(default=None, description="The damage type of the economic impact caused by the damaged infrastructure (qualitative information).")
#     impact_origin: Literal["repair_recovery_cost","downstream_economic_consequence"] = Field(
#         description=("Whether the impact is a direct cost of repairing, restoring, replacing or recovering damaged infrastructure, or a downstream economic consequence caused by infrastructure damage or disruption."
#     ))
#     # magnitude: str | None = Field(default=None, description="Monetary consequence of the damaged infrastructure.")
#     amount: str | None = Field(default=None, description="Amount of the economic consequence (quantitative information).") # e.g. thousands of people, 3,000 households
#     unit: Literal[None, "*€*", "*EURO*", "Euros", "£",] | None = Field(default=None, description="Currency or Unit of the economic consequence (qualitative information).")


class SocietalImpact(BaseModel):

    source_asset: str  = Field(
        description=("Infrastructure asset or service whose damage or disruption caused the societal impact."
    ))
    source_location: str | None = Field(
        default=None,
        description=("Location of the infrastructure asset associated with the societal impact."
    ))
    impact_origin: Literal[None, "evacuation_or_displacement", "loss_of_critical_service_or_inaccessibility", "disruption_of_mobility", "other_affected_population"] = Field(
        default=None,
        description=("Category of the societal impact: "
            "'evacuation_or_displacement' (people or households evacuated, displaced, relocated or made homeless); "
            "'loss_of_critical_service_or_inaccessibility' (people, households or communities without electricity, water, gas, heating or telecommunication, without access to healthcare, schools or other essential services or facilities, or communities isolated or cut off); "
            "'disruption_of_mobility' (people, passengers or commuters unable to travel, stranded or with disrupted journeys); "
            "'other_affected_population' (any other explicitly reported consequence for people, households or communities, e.g. generally affected people or disrupted living conditions, missing people, fatalities)."
    ))
    # tool calling does not enforce the Literal: e.g. 'power_outage' -> 'loss_of_critical_service_or_inaccessibility'
    _map_impact_origin = field_validator("impact_origin", mode="before")(
        lambda v: coerce_category(v, SOCIETAL_ORIGIN_KEYWORDS, fallback="other_affected_population"))
    impact_type: str | None = Field(
        description=("Specific type of societal impact, for example: affected households / families / people / passengers / commuters / students or pupils / population /community / residential area, who are displaced, affected by loss or access to essential services or by disruption of mobility."
    ))
    # keep only human-related impact types: e.g. 'evacuated residents' -> 'people', 'school pupils' -> 'students', 'power outage' -> None
    _map_impact_type = field_validator("impact_type", mode="before")(
        lambda v: coerce_category(v, SOCIETAL_TYPE_KEYWORDS, fallback=None))
    # damage_type: str | None = Field(
    #     default=None,
    #     description=("Original damage, failure, or disruption affecting the source infrastructure."
    # ))
    # description: str = Field(
    #     description=("Concise description of the societal impact explicitly reported in the text."
    #     ))
    amount: str | None = Field(
        default=None,
        description=("Quantitative magnitude explicitly stated in the text, if available. Examples: 'up to 300', 'approximately 10,000', '1.000', 'around 10 000'. Do not calculate or infer values."
        ))
    unit: str | None = Field(
        default=None,
        description=("Unit associated with the quantitative amount, if explicitly stated. Examples: 'households', 'people', 'families', 'passengers', 'patients', 'students', 'residents', 'citizens',  'commuters', 'farmers', 'communities', 'municipalities'."
        )
    )


# class SocietalImpact(BaseModel):
#     source_asset: str | None = Field(default=None, description="Infrastructure asset or service whose damage or disruption caused the societal impact.")
#     source_location: str | None = Field(default=None, description="Original location of the source asset.")
#     impact_type: str | None = Field(default=None, description="The type of the societal impact caused by the damaged infrastructure.")
#     damage_type: str | None = Field(default=None, description="The damage type of the societal impact caused by the damaged infrastructure (qualitative information).")
#     # impact_description: str = Field(description="Non-monetary societal consequence.")
#     magnitude: str | None = Field(default=None, description="Quantitative magnitude of the societal impact, e.g. '5,000 households', '3 school days', '20.000 people'.")
#     amount: str | None = Field(default=None, description="Amount of the economic consequence.")
#     unit: Literal[None, "*families*", "*households*", "*people*", "*passengers*", "*patients*", 
#         "*students*", "*pupils*", "*residents*", "*citizens*",  "*commuters*", "*farmers*", "*communities*", "municipalities"
#     ] | None = Field(default=None, description="Currency or Unit of the economic consequence.")


class EnvironmentalImpact(BaseModel):

    source_asset: str = Field(
        description=("Infrastructure asset or service whose damage or disruption caused the environmental impact."
    ))
    source_location: str | None = Field(
        default=None,
        description=("Location of the infrastructure asset associated with the environmental impact."
    ))
    impact_origin: Literal[None, "environmental_impact"] = Field(
        default=None,
        description=("Category of the impact: 'environmental_impact' (any explicitly reported consequence for the natural environment, e.g. land, soil, groundwater, rivers or regions)."
    ))
    # tool calling does not enforce the Literal: only one category for now, any reported origin -> 'environmental_impact'
    _map_impact_origin = field_validator("impact_origin", mode="before")(
        lambda v: None if v is None else "environmental_impact")
    impact_type: str | None = Field(
        default=None,
        description=("Specific type of environmental impact, for example: "
            "'waste fills and debris' (waste, debris or rubble deposited on land, in or along rivers); "
            "'inundated' (flooded or inundated land, fields, areas, or regions); "
            "'pollution of environment' (land, soil, groundwater, rivers etc.)"  # polluted by oil, diesel, chemicals, sewage etc.)."
    ))
    amount: str | None = Field(
        default=None,
        description=("Quantitative magnitude explicitly stated in the text, if available. Preserve the original wording and do not calculate or infer values. Examples: '15 cm thick', 'around 100 hectares', 'more than 2,000 tonnes', '7 out of 31'."
    ))
    unit: str | None = Field(
        default=None,
        description=("Unit associated with the quantitative amount, if explicitly stated. Examples: 'cm', 'hectares', 'square kilometres', 'km²', 'tonnes', 'litres', 'cubic metres'."
    ))


class SecondaryInfrastructureImpact(BaseModel):
    source_asset: str = Field(description="Infrastructure asset that was originally damaged or disrupted.")
    source_location: str | None = Field(default=None, description="Original location of the source asset.")
    source_damage: str | None = Field(default=None, description="Original damage or disruption affecting the source asset.")
    target_asset: str | None = Field(default=None, description="Other infrastructure asset or infrastructure-related service affected as a consequence of the source damage.")
    # e.g. closure of 3 school days
    target_damage: Literal[
        None, "structural_damage", "inundation", "contamination", "service_loss", "obstruction", "disruption", "affected", "restored_recovered"
    ] | None = Field(
        default=None,
        description=(
            "General category of the resulting damage, disruption or operational effect experienced by the target infrastructure asset or service. "
            "If several categories apply, use the most severe one (the first one in this list):\n" + pp.describe_damage_categories()
        ),
    )
    # impact_description: str = Field(description="Explicitly stated downstream impact on the infrastructure asset or service.")

    # tool calling does not enforce the Literal: map free-text damage (e.g. "severely damaged") to its category, unknown -> None
    _map_target_damage = field_validator("target_damage", mode="before")(lambda v: pp.map_damage_to_category(v))


# class SecondaryInfrastructureImpact(BaseModel):

#     source_asset: str = Field(
#         description=("Infrastructure asset or service originally damaged or disrupted.")
#     )
#     source_location: str | None = Field(
#         default=None,
#         description=("Location of the original infrastructure asset.")
#     )
#     source_damage: str | None = Field(
#         default=None,
#         description=("Original damage, failure, or disruption affecting the source infrastructure."
#         ))
#     impact_type: str = Field(
#         description=(
#             "Type of secondary infrastructure impact, for example: service disruption, outage, closure, delay, reduced capacity, restricted access, interrupted supply, or loss of connectivity."
#     ))
#     target_asset: str = Field(
#         description=("Other infrastructure asset or infrastructure-related service affected as a consequence of the source damage.")
#     )
#     target_damage: str | None = Field(
#         default=None, 
#         description=("The resulting condition, disruption, or operational effect experienced by the affected infrastructure asset or service. Examples: delayed, disrupted, closed, blocked, unavailable, reduced capacity, inoperable, restricted access, interrupted, failed, loss of connectivity, interrupted supply."
#     ))


class InfrastructureCase(BaseModel):

    is_infrastructure_case: bool
    infrastructure_type: str =  Field(
        default=None,
        description="Infrastructure assets or services damaged by a natural hazard.")
    infrastructure_amount: str | None = Field(
        default=None,
        description=("Number of affected infrastructure assets or services (infrastructure_type) explicitly reported in the text, if any. Preserve the original wording. Example: In the villages around Berlin around 100 roads were damaged. infrastructure_type:'roads', infrastructure_amount:'around 100'"
    ))
    infrastructure_name: str | None = Field(
        default=None,
        description=("Proper name or designation of the affected infrastructure asset (infrastructure_type) explicitly mentioned in the text, if any. Preserve the original wording. "
                     "Examples: 'A1', 'B510', 'Gates Avenue', 'Braunschweig chemical plant', 'Berlusconi Airport'. "
                     "Example: The A1 highway was flooded near Cologne. infrastructure_type:'A1 highway', infrastructure_name:'A1'"
    ))
    location: str | None = Field(
        default=None,
        description="Location explicitly associated with this infrastructure asset."
    )
    # damage: Literal[
    #     None, "affected", "damaged", "severely damaged", "highly damaged", "disrupted", "blocked", "destroyed", "largely destroyed", 
    #     "heavily destroyed", "partly closed", "closed", "outage(s)", "blackout", "loss of water/electricity", "no supply", "left without water/electricity", "contaminated", "polluted", "flooded", "inundated", 
    #     "fallen tree(s)", "broken tree(s)", "overturned tree(s)", "uprooted tree(s)", "felled tree(s)", "debris", "debris removal", 
    #     "debris on", "water on", "dam failure(s)", "dyke break(s)", "dike break(s)", "breached dike(s)", "broken embankment(s)", 
    #     "closure(s)", "derailed", "impassable", "traffic jam(s)", "traffic disruption(s)", "traffic congestion(s)", 
    #     "indirectly affected", "flight delay(s)", "train delay(s)", "delay(s)", "train failure(s)", "failure(s)", "cancellation(s)", 
    #     "canceled", "re-established", "reestablished", "restored", "reconnected", "rebuild", "repaired"
    # ] | None = Field(
    #     default=None,
    #     description="Damage, failure, disruption, or operational status explicitly associated with this infrastructure asset."
    # )
    damage: Literal[
        None, "structural_damage", "inundation", "contamination", "service_loss", "obstruction", "disruption", "affected", "restored_recovered"
    ] | None = Field(
        default=None,
        description=(
            "General category of the damage, failure, disruption, or operational status explicitly associated with this infrastructure asset. "
            "If several categories apply, use the most severe one (the first one in this list):\n" + pp.describe_damage_categories()
        ),
    )

    # tool calling does not enforce the Literal: map free-text damage (e.g. "severely damaged") to its category, unknown -> None
    _map_damage = field_validator("damage", mode="before")(lambda v: pp.map_damage_to_category(v))

   
# damage categories of the schema must be the same as in src/postprocess.py (DAMAGE_CATEGORIES)
assert [c for c in typing.get_args(typing.get_args(InfrastructureCase.model_fields["damage"].annotation)[0]) if c] == list(pp.DAMAGE_CATEGORIES)
assert [c for c in typing.get_args(typing.get_args(SecondaryInfrastructureImpact.model_fields["target_damage"].annotation)[0]) if c] == list(pp.DAMAGE_CATEGORIES)


class InfrastructureCases(BaseModel):
    cases: list[InfrastructureCase]
   

class GraphState(MessagesState): # TypedDict
    text: str
    # chunk_relevant: bool
    # assets: list[str]
    # damages: list[str]
    # locations: list[str]
    is_infrastructure_case: bool
    assets_org: InfrastructureCases
    infrastructure_cases: InfrastructureCases
    
    # # assets: Annotated[list, add_messages]

    # # extracted_cases: list[InfrastructureCase] # List[Dict[str, str]]
    # # verified_cases: list[VerifiedInfrastructureCase]
    # # final_output: List[Dict]

    # cascades
    has_cascading_impacts: bool
    economic_impacts: List[EconomicImpact]
    secondary_infrastructure_impacts: List[SecondaryInfrastructureImpact]
    societal_impacts: List[SocietalImpact]
    environmental_impacts: List[EnvironmentalImpact]


# record column -> field of the extraction schema (pydantic class) whose definition is given to the judge
JUDGE_FIELD_SCHEMA = {
    "infrastructure_type": (InfrastructureCase, "infrastructure_type"),
    "infrastructure_amount": (InfrastructureCase, "infrastructure_amount"),
    "infrastructure_name": (InfrastructureCase, "infrastructure_name"),
    "location": (InfrastructureCase, "location"),
    "damage": (InfrastructureCase, "damage"),
    "economic_impact_origin": (EconomicImpact, "impact_origin"),
    "economic_impact_type": (EconomicImpact, "impact_type"),
    "economic_impact_amount": (EconomicImpact, "amount"),
    "economic_impact_unit": (EconomicImpact, "unit"),
    "societal_impact_origin": (SocietalImpact, "impact_origin"),
    "societal_impact_type": (SocietalImpact, "impact_type"),
    "societal_impact_amount": (SocietalImpact, "amount"),
    "societal_impact_unit": (SocietalImpact, "unit"),
    "environmental_impact_origin": (EnvironmentalImpact, "impact_origin"),
    "environmental_impact_type": (EnvironmentalImpact, "impact_type"),
    "environmental_impact_amount": (EnvironmentalImpact, "amount"),
    "environmental_impact_unit": (EnvironmentalImpact, "unit"),
    "secondary_infrastructure_asset": (SecondaryInfrastructureImpact, "target_asset"),
}

# derived columns (CI grouping), never shown to or adapted by the judge
JUDGE_EXCLUDED_FIELDS = ["infrastructure_group", "infrastructure_sector", "secondary_infrastructure_group", "secondary_infrastructure_sector"]
assert not set(JUDGE_FIELD_SCHEMA) & set(JUDGE_EXCLUDED_FIELDS)

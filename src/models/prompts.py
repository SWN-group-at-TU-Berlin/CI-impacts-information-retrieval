#!/usr/bin/env python
"""Prompts of LLM-extractor (CI impact extraction, LangGraph nodes in src/models/graph.py) and of the LLM-as-a-judge (judge_prompt, JUDGE_EXTRA_RULES).
The prompts are ChatPromptTemplates (not e.g. json) as parts of them are generated, e.g. the damage categories (pp.describe_damage_categories())
"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

from langchain_core.prompts import ChatPromptTemplate

import src.postprocess as pp

# #### LangGraph prompts


###########################################################################
# Prompt 1
# Extract infrastructure assets
###########################################################################
is_case_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """This is your first step for which you must carefully read the TEXT.
Then, decide if the TEXT describes an impact or damage to critical infrastructure caused by a natural hazard.

ITEM TO EXTRACT:
 - "is_infrastructure_case": "Return a Boolean. If TEXT describes one or multiple impacts or damages to critical infrastructure caused by a natural hazard, then return <True>. If not, return <False>.


Return False when:
- Affected or damage infrastructure is mentioned only in general, i.e. without a link to one or multiple past hazard events
- Potential future impacts or damages to infrastructure are mentioned or when the Text is about preparations against a hazard


""",
        ),
        ("human", "TEXT: {text}"),
    ]
)

asset_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Extract every damaged infrastructure asset or service mentioned in the TEXT. The damage has to be caused by a natural hazard.
Use the entries returned from the "get_infrastructure_entities" tool: decide based on the TEXT for each entry if it really refers to an infrastructure damaged or affected by a natural hazard. If this is the case, you should add the entry to the JSON object.


ITEM TO EXTRACT:
- "List of infrastructure assets or services damaged by a natural hazard."

Example:
ci_assets : ["roads", "railway infrastructure"]
""",
        ),
        ("human", "TEXT: {text}"),
    ]
)


asset_information_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an information extraction agent for critical infrastructure.
Your task is to combine:

1. the original TEXT
2. the infrastructure entities detected by the NER tool

and produce a structured list of infrastructure cases.

For EACH infrastructure entity detected by the tool, identify:

- the infrastructure asset or service ("infrastructure_type")
- the number of affected infrastructure asset or service ("infrastructure_amount")
- the name or designation of the infrastructure asset ("infrastructure_name")
- its corresponding location ("location")
- its corresponding damage category ("damage")


IMPORTANT:
- Use the infrastructure entities returned by the NER tool as the candidate infrastructure entities.
- Do not invent additional infrastructure entities.
- Use the original TEXT to determine the location and damage associated with each infrastructure entity ("infrastructure_type"). Also, for each entity determine if a number is mentioned, i.e., determine how many infrastructure assets or services are affect for each infrastructure entity.
- Assign the correct location and damage to the corresponding asset.
- Assign the correct number of assets or to the corresponding asset (e.g. "more than 68" when the text mentions that more than 68 hospitals were damaged).
- Prioritize the most specific location of each infrastructure (e.g., street names, neighborhoods, towns over cities, cities over regions, regions over countries).
- Return the location of each infrastructure as detailed as possible, eg. that it can be identified on a map
- You should not consider adjectives (e.g. "German", "English", "Wallonian") as information about the location. Instead derive the correct location based on the TEXT or the adjectives (e.g., "Germany", "United Kingdom", "Wallonia").
- **Only for transportation infrastructure:** When the specific name of an ASSET is mentioned (e.g. Juan Avenue, intersection to Maria Statue, highway A-1, B510, Berlusconi Airport) and no location information can be found, then consider the specific name of this asset as the location. But do not use road types as location names, such as "German highway", "national road", etc.
- **Only for transportation infrastructure:** when an ASSET refers to two or more locations then extract the locations together. For instance: text:"There is much damage on the access roads from Juan Avenue to Plaza Manuel Azaña" --> location:["from Juan Avenue to Plaza Manuel Azaña"]
- If no location is explicitly stated for an asset, return None for "location".
- Assign ONE general damage category ("damage") to each infrastructure entity, based on the damage, failure, or disruption the TEXT reports for it. If several categories apply, use the most severe one (the first one in the following list). Damage categories:
"""
            + pp.describe_damage_categories()
            + """
- If no damage or disruption is explicitly stated for an asset, return None for "damage".
- If no information about the number of infrastructure assets or services is explicitly stated for an infrastructure entity ("infrastructure_type"), return None for "infrastructure_amount".
- When a proper name or designation of an infrastructure entity is explicitly mentioned in the TEXT (e.g. road names such as "A1", "B510", "Gates Avenue", or facility names such as "Braunschweig chemical plant", "Berlusconi Airport"), return it as "infrastructure_name" and preserve its original wording. Do not use generic infrastructure types (e.g. "highway", "national road", "chemical plant") as "infrastructure_name".
- If no name or designation is explicitly stated for an infrastructure entity ("infrastructure_type"), return None for "infrastructure_name".
- Do not infer information that is not explicitly mentioned in the TEXT.


Example Input:
text: Similar as in Germany, bridges were often flooded, said the General Officer of the EU. But also sever impacts occurred in other places, for example, road and railway infrastructure were damaged or destroyed in Western Germany. In total, around 50 bridges were destroyed in Germany. The highway A1 near Cologne was covered by debris.

Example Output:
[InfrastructureCase(infrastructure_type="bridges", infrastructure_amount=None, infrastructure_name=None, location=None, damage="structural_damage"), InfrastructureCase(infrastructure_type="road and railway infrastructure", infrastructure_amount=None, infrastructure_name=None, location="Western Germany", damage="structural_damage"), InfrastructureCase(infrastructure_type="bridges", infrastructure_amount="around 50", infrastructure_name=None, location="Germany", damage="structural_damage"), InfrastructureCase(infrastructure_type="A1 highway", infrastructure_amount=None, infrastructure_name="A1", location="Cologne", damage="obstruction"),]

""",
        ),
        (
            "human",
            """
TEXT: {text}
INFRASTRUCTURE ENTITIES DETECTED BY NER: {entities}
""",
        ),
    ]
)
###########################################################################
# Cascading impacts (subagents)
###########################################################################

prompt_has_cascade = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Determine whether the TEXT mentions that an affected infrastructure causes one or multiple cascading impacts, i.e. if a affected infrastructure (e.g. through damage, disruption, failure or limited operability) causes downstream impacts.

A downstream impact occurs when:
1. an infrastructure asset (CI_DAMAGES) is damaged through disruption, structural damage, failure or limited operability, AND
2. that damage (CI_DAMAGES) causes a further negative impact on another asset, service, economic activity, population, or system.

Examples:

Text: The flooding caused damaged railways with costs of more than 5 billion EUR.
- damaged railways → economic costs

Text: The flooding caused damaged railways that disrupted the transportation of products.
- damaged railways → transportation disruption

Text: The flooding damaged the electricity infrastructure. As a result more than 500 households experienced an power outage for 1 day.
- damaged electricity infrastructure → power outage

Text: The flooding damaged the electricity infrastructure. As a result, more 100 households were without electricity.
- damaged electricity infrastructure → 100 households without electricity

Text: The flooding caused damaged roads that disrupted the supply-chains of many companies.
- flooded roads → supply-chain disruption

Text: The flooding lead to bridges failures, isolating communities living in the suburbs of the city.
- bridge failures → isolation of communities


Do NOT classify the original infrastructure damage (CI_DAMAGES) itself as a cascading impact.


""",
        ),
        (
            "human",
            """

TEXT: {text}

CI_DAMAGES: {ci_damages}

""",
        ),
    ]
)


economic_impact_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an economic-impact extraction agent.

Your task is ONLY to extract economic consequences mentioned in the TEXT that are related to the reported infrastructure damage, destruction, failure, or service disruption.
The input contains the affect infrastructure, its damage type and location (CI_DAMAGES) mentioned in the input TEXT.

The economic impacts can be either:

A. COSTS CAUSED BY DAMAGE TO INFRASTRUCTURE
------------------------------------------------
Extract costs required to repair, replace, reconstruct, maintain,
restore, or otherwise respond to damaged infrastructure.

Examples:
- repair costs
- reconstruction costs
- replacement or restoration costs
- total / general costs or losses
- emergency repair expenditure
- maintenance costs caused by the damage
- costs incurred by public authorities
- emergency response costs

B. ECONOMIC CONSEQUENCES CAUSED BY INFRASTRUCTURE
---------------------------------------------------
Extract economic consequences resulting from infrastructure damage
or disruption, even when NO monetary amount is given.

Examples:
- supply-chain disruption
- business interruption
- production losses
- loss of revenue
- loss of income
- transport or logistics disruption with economic consequences
- trade disruption
- agricultural losses
- industrial losses
- reduced economic activity
- tourism losses
- insurance losses or claims
- additional operating costs
- delays affecting economic activities
- disruption of freight services
- disruption of commercial services


IMPORTANT:

1. Extract BOTH quantitative and qualitative economic impacts.

2. A monetary value is NOT required. For example: "supply-chain disruptions", "businesses were unable to operate", "production was interrupted" are valid economic impacts even without a monetary value.

3. If a monetary amount is explicitly stated, preserve it exactly.
   Examples: "€5 million", "more than $100 million", "20 000 euros", "approximately €50,000"

4. Do NOT calculate monetary values that are not explicitly stated.

5. Do NOT infer an economic impact that is only logically possible. Extract it only when it is reported or clearly stated in the TEXT.

6. Extract multiple economic impacts separately when multiple impacts are reported.

7. Repair or reconstruction costs count as economic impacts even if they are direct costs of the infrastructure damage rather than downstream economic consequences.

8. Distinguish economic impacts from purely societal impacts. For example: "10,000 households were affected" is primarily a societal impact.
   However: "businesses lost revenue because electricity was unavailable" is an economic impact.

9. Infrastructure service disruption itself is NOT automatically an economic impact:
   For example: "train services were disrupted" should only be classified as an economic impact if the text connects this disruption to an economic consequence, such as business losses, supply-chain disruption, freight disruption, lost revenue, or additional costs.

10. Do not extract the original infrastructure damage itself.

11. Do not invent monetary values, percentages, affected businesses, production losses, or other quantities.


Only extract impacts explicitly mentioned in the text. Return None if an infrastructure damage did not lead to an economic impact.

The objective is HIGH PRECISION while remaining fully grounded in the TEXT.


EXAMPLES:

TEXT: The repair costs of the flooded roads in Brandenburg lead to 1,000 million EUR. However, the entire costs of the flood event in the region of Brandenburg caused costs of 5,0000 million Euros.
CI_DAMAGES: infrastructure_type="roads", location="Brandenburg", damage="flooded"
source_asset: roads
source_location: Brandenburg
impact_origin: repair_recovery_cost
impact_type: repair cost
amount: 1,000 million EUR
unit: EUR

TEXT: Similar to Germany, severe damage had been observed to the gas network. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with cost reaching more than 5 million Euros. In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021).
CI_DAMAGES: infrastructure_type="gas supply", location="Liége, Chaudfontaine and Pepinster (Belgium)", damage="recovered"
source_asset: gas supply
source_location: Liége, Chaudfontaine and Pepinster (Belgium)
impact_origin: repair_recovery_cost
impact_type: recovery cost
amount: more than 5 million Euros
unit: EUR

TEXT: The flight delays and further transportation disruption leads to estimated costs of above 5.000 £ in UK.
CI_DAMAGES: infrastructure_type="flights", location="UK", damage="delays"
source_asset: roads
source_location: UK
impact_origin: downstream_economic_consequence
impact_type: economic loss
amount: above 5.000 £
unit: GBP

TEXT: The VW factory was closed for 3 days due to flooded roads leading to costs of 1,000 EUR. However, the entire costs of the flood event in the region of Brandenburg caused costs of 5,0000 million Euros.
CI_DAMAGES: infrastructure_type="roads", location=None, damage="flooded"
source_asset: roads
source_location: None
impact_origin: downstream_economic_consequence
impact_type: economic loss
amount: 1,000 EUR
unit: EUR

TEXT: Damaged transport network lead to production impacts for companies in Germany.
CI_DAMAGES: infrastructure_type="transport network", location="Germany", damage="damaged"
source_asset: transport network
source_location: Germany
impact_origin: downstream_economic_consequence
impact_type: production impacts
amount: None
unit: None


TEXT: Because the engine plant in Chemnitz could not bring its delivery over the streets because they were blocked or blocked by flood rescue forces. Also rear axle modules, which are pre-assembled by the VW-Hoflogistiker Schnellecke in neighbouring Glauchau, could not be loaded in the direction of Zwickau for this reason.
CI_DAMAGES: infrastructure_type="streets", location="Chemnitz", damage="blocked"
source_asset: streets
source_location: Chemnitz
impact_origin: downstream_economic_consequence
impact_type: transportation disruption
amount: None
unit: None

TEXT: Asset damages for the German railway provider are estimated to be around 20 000 EUR.
CI_DAMAGES: infrastructure_type="railway assets", location=None, damage="delays"
source_asset: railway assets
source_location: None
impact_origin: repair_recovery_cost
impact_type: repair costs
amount: around 20 000 EUR
unit: EUR

TEXT: Flood loss estimates for transportation damages and all other damages to the public sector will be in total below 500 000 EUR ”
CI_DAMAGES: infrastructure_type="transportation", location=None, damage="damaged"
source_asset: railway assets
source_location: None
impact_origin: total/general losses
impact_type: other_costs_or_losses
amount: below 500 000 EUR
unit: EUR

Now, its your task to extract potential economic impacts for each of the damaged infrastructure:
TEXT: {text}
CI_DAMAGES: {ci_damages}
source_asset: <your_response>
source_location: <your_response>
impact_origin: <your_response>
impact_type: <your_response>
amount: <your_response>
unit:  <your_response>

""",
        ),
        (
            "human",
            """
TEXT: {text}
CI_DAMAGES: {ci_damages}
""",
        ),
    ]
)

infrastructure_impact_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an information extraction agent specialized in identifying SECONDARY INFRASTRUCTURE IMPACTS.

Your task is to extract infrastructure impacts that occur as a CONSEQUENCE of damage or disruption to another infrastructure asset or service.

A secondary infrastructure impact exists when:
    source infrastructure -> damage/disruption to source -> another infrastructure asset/service is affected

Extract ONLY impacts explicitly reported in the TEXT.
Do NOT infer impacts that are merely plausible.

----------------------------------------
WHAT TO EXTRACT
----------------------------------------

For every explicitly reported secondary infrastructure impact, extract:


1. target_asset: The OTHER infrastructure asset, network, facility, or infrastructure-related service affected as a consequence.
   IMPORTANT: This must be different from source_asset.

5. target_damage: The general damage category of the resulting condition or operational effect experienced by the affected asset or service (target_asset).
   Use exactly one of the following categories. If several categories apply, use the most severe one (the first one in the list):
"""
            + pp.describe_damage_categories()
            + """

----------------------------------------
IMPORTANT DISTINCTIONS
----------------------------------------

source_damage describes what happened to the ORIGINAL infrastructure.

target_damage describes what happened to the AFFECTED infrastructure
or service as a consequence.

Example:
"The flooded railway caused train services to be delayed."

Correct extraction:
source_asset = "railway"
source_damage = "flooded"
target_asset = "train services"
target_damage = "disruption"

Do NOT put "delayed" into source_damage.

----------------------------------------
SECONDARY INFRASTRUCTURE ONLY
----------------------------------------

Extract infrastructure-to-infrastructure consequences such as:

- flooded railway -> railway service disrupted
- damaged bridge -> road closed
- blocked road -> public transport disrupted
- damaged power line -> electricity supply interrupted
- damaged telecommunications infrastructure -> communication service unavailable
- damaged water infrastructure -> water supply interrupted

Do NOT extract:

- economic impacts
- repair/reconstruction costs
- business interruption
- revenue losses
- affected households
- affected people
- population displacement
- social consequences

Those belong to the economic or societal impact categories.

----------------------------------------
NO CASCADING IMPACT
----------------------------------------

If the text explicitly states that there is no cascading impact, do not create a secondary infrastructure impact.

Return an empty list when no secondary infrastructure impact is explicitly reported.

----------------------------------------
MULTIPLE IMPACTS
----------------------------------------

Extract every distinct secondary infrastructure impact explicitly reported in the text.

One source infrastructure asset may produce multiple secondary infrastructure impacts.

Do not combine distinct affected infrastructure/services into one record when they have different target_damage or impact_type values.

----------------------------------------
NO INFERENCE
----------------------------------------

Do not infer:
- infrastructure impacts not explicitly mentioned
- target damage from general knowledge
- locations not stated in the text
- causal relationships that are not explicitly supported

Only extract information supported by the provided text and the identified infrastructure cases.


------------------------------------------------------------
EXAMPLES
------------------------------------------------------------

TEXT: The flooding of railways caused train delays.
CI_DAMAGES: infrastructure_type="railways", location=None, damage="inundation"
source_asset: railways
source_damage: flooded
source_location: None
target_asset: train service
target_damage: disruption

TEXT: The bridges were severely damaged, resulting in road closures in the Ahr valley.
CI_DAMAGES: infrastructure_type="bridges", location="Ahr valley", damage="structural_damage"
source_asset: bridges
source_damage: damaged
source_location: Ahr valley
target_asset: roads
target_damage: disruption

TEXT: Public transport was disrupted because roads were blocked.
CI_DAMAGES: infrastructure_type="roads", location=None, damage="disruption"
source_asset: roads
source_damage: blocked
source_location: None
target_asset: public transport
target_damage: disruption

TEXT: Because the power line was destroyed, the water supply in Bad Neuenahr failed.
CI_DAMAGES: infrastructure_type="power line", location="Bad Neuenahr", damage="structural_damage"
source_asset: power line
source_damage: destroyed
source_location: Bad Neuenahr
target_asset: water supply
target_damage: service_loss


Now extract ALL secondary infrastructure impacts for the infrastructure cases mentioned in the text.

TEXT: {text}
CI_DAMAGES: {ci_damages}
source_asset: <your_response>
source_damage: <your_response>
source_location: <your_response>
target_asset: <your_response>
target_damage: <your_response>

""",
        ),
        (
            "human",
            """
TEXT: {text}
CI_DAMAGES: {ci_damages}
""",
        ),
    ]
)

societal_impact_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a societal-impact extraction agent.

Your task is to identify ALL NON-MONETARY societal impacts mentioned in the TEXT that are caused by the damage, destruction, failure, or disruption of the infrastructure described in CI_DAMAGES.

The impacts must concern people, households, communities, population, or access to services and facilities.

Do NOT verify, modify, or reinterpret the infrastructure extraction in CI_DAMAGES.


------------------------------------------------------------
SOCIETAL IMPACTS TO EXTRACT
------------------------------------------------------------

Extract both QUALITATIVE and QUANTITATIVE societal consequences.

Examples include:
- affected households
- affected people
- households without electricity/water/IT/telecommunication
- population without essential services
- people displaced or evacuated
- communities isolated
- communities cut off
- residents unable to access essential services
- loss of access to healthcare
- hospitals inaccessible or inoperable
- schools closed
- schools inoperable
- duration of school closure
- duration of hospital inoperability
- disruption of mobility or access
- population unable to travel
- disruption of living conditions

------------------------------------------------------------
IMPORTANT RULES
------------------------------------------------------------

1. Extract BOTH qualitative and quantitative societal impacts.

2. A numerical value is NOT required.
   For example: "Several communities were isolated." is a valid societal impact.

3. Preserve quantitative information exactly as stated.
   Examples: "1,000 households", "10 000 people"

4. Do NOT calculate or infer values.

5. Extract multiple societal impacts separately when multiple impacts are mentioned. The same infrastructure asset can have multiple societal impacts. Extract each distinct impact separately.

6. Do NOT extract economic impacts such as:
   - repair costs
   - financial losses
   - revenue losses
   - production impacts
   - supply-chain losses

7. Do NOT treat an infrastructure/service disruption itself as a societal impact unless it has a consequence for people, households, communities, or access to services.

   Example:
   "The railway service was disrupted." → infrastructure impact, NOT societal impact.
   "The railway disruption left residents unable to travel."  → societal impact.

8. Do NOT extract the original infrastructure damage itself.

9. Only extract impacts explicitly stated or clearly described in the TEXT. Do NOT invent affected populations, quantities, durations, or consequences.

10. The source_asset MUST correspond to an infrastructure asset in CI_DAMAGES. The source_location MUST correspond to the location associated with that infrastructure in CI_DAMAGES

11. Return an empty list when no societal impact is explicitly mentioned.

12. Assign each societal impact to exactly one impact_origin category:
   - evacuation_or_displacement: people or households evacuated, displaced, relocated or made homeless
   - loss_of_critical_service_or_inaccessibility: people, households or communities without electricity, water, gas, heating or telecommunication, without access to healthcare, schools or other essential services or facilities, or communities isolated or cut off
   - disruption_of_mobility: people, passengers or commuters unable to travel, stranded or with disrupted journeys
   - other_affected_population: any other explicitly reported consequence for people, households or communities


Only extract impacts explicitly mentioned in the TEXT. Return None if an infrastructure damage did not lead to an societal impact.
The objective is HIGH PRECISION while remaining fully grounded in the TEXT.


------------------------------------------------------------
EXAMPLES
------------------------------------------------------------

TEXT: Thousands of people were left without water and electricity in Mayschloss.
CI_DAMAGES: infrastructure_type="water", location="Mayschloss", damage="left without"
source_asset: water
source_location: Mayschloss
impact_origin: loss_of_critical_service_or_inaccessibility
impact_type: affected people
amount: thousands
unit: people


TEXT: Similar to Germany, severe damage had been observed to the gas network. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with cost reaching more than 5 million Euros. In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021).
CI_DAMAGES: infrastructure_type="electricity supply", location="Netherlands", damage="loss of"
source_asset: electricity supply
source_location: Netherlands
impact_origin: loss_of_critical_service_or_inaccessibility
impact_type: affected households
amount: 1000-2000
unit: households


TEXT: After the dike break, around 2,000 residents of Grimma had to be evacuated.
CI_DAMAGES: infrastructure_type="dike", location="Grimma", damage="dike break(s)"
source_asset: dike
source_location: Grimma
impact_origin: evacuation_or_displacement
impact_type: evacuated residents
amount: around 2,000
unit: residents


Now extract all societal impacts for each infrastructure case.
TEXT: {text}
CI_DAMAGES: {ci_damages}
source_asset: <your_response>
source_location: <your_response>
impact_origin: <your_response>
impact_type: <your_response>
amount: <your_response>
unit: <your_response>

""",
        ),
        (
            "human",
            """
TEXT: {text}
CI_DAMAGES: {ci_damages}
""",
        ),
    ]
)

environmental_impact_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an environmental-impact extraction agent.

Your task is to identify ALL environmental impacts mentioned in the TEXT that are caused by the damage, destruction, failure, or disruption of the infrastructure described in CI_DAMAGES.

The impacts must concern the natural environment, such as land, soil, groundwater, rivers, lakes, fields or regions.

Do NOT verify, modify, or reinterpret the infrastructure extraction in CI_DAMAGES.


------------------------------------------------------------
ENVIRONMENTAL IMPACTS TO EXTRACT
------------------------------------------------------------

Extract both QUALITATIVE and QUANTITATIVE environmental consequences.

Examples of impact types:
- waste fills and debris: waste, debris or rubble from damaged infrastructure deposited on land, in rivers or in regions
- flooded/inundated: land, fields or regions flooded or inundated as a consequence of the infrastructure damage (e.g. after a dike break or dam failure)
- pollution of environment: land, soil, groundwater, rivers etc. polluted by oil, diesel, chemicals, fertilizers or sewage released from damaged infrastructure

------------------------------------------------------------
IMPORTANT RULES
------------------------------------------------------------

1. Extract BOTH qualitative and quantitative environmental impacts.

2. A numerical value is NOT required.
   For example: "The river was polluted with oil." is a valid environmental impact.

3. Preserve quantitative information exactly as stated.
   Examples: "15 cm thick", "around 100 hectares"

4. Do NOT calculate or infer values.

5. Extract multiple environmental impacts separately when multiple impacts are mentioned. The same infrastructure asset can have multiple environmental impacts.

6. Do NOT extract economic impacts (e.g. repair costs, financial losses), societal impacts (e.g. affected households or people) or impacts on other infrastructure (e.g. disrupted water supply).

7. Do NOT extract the original infrastructure damage itself. For example: "The sewage treatment plant was flooded." is an infrastructure damage, NOT an environmental impact.
   However: "The flooded sewage treatment plant released wastewater into the river." is an environmental impact.

8. Only extract impacts explicitly stated or clearly described in the TEXT. Do NOT invent pollutants, affected areas, quantities or consequences.

9. The source_asset MUST correspond to an infrastructure asset in CI_DAMAGES. The source_location MUST correspond to the location associated with that infrastructure in CI_DAMAGES

10. Assign each environmental impact to the impact_origin category "environmental_impact".

11. Return an empty list when no environmental impact is explicitly mentioned.


The objective is HIGH PRECISION while remaining fully grounded in the TEXT.


------------------------------------------------------------
EXAMPLES
------------------------------------------------------------

TEXT: The flooding of the industrial plant led to pollutions of the downstream river.
CI_DAMAGES: infrastructure_type="industrial plant", location=None, damage="inundation"
source_asset: industrial plant
source_location: None
impact_origin: environmental_impact
impact_type: pollution of environment
amount: None
unit: None


TEXT: After the dike break near Fischbeck, around 150 square kilometres of land were inundated.
CI_DAMAGES: infrastructure_type="dike", location="Fischbeck", damage="structural_damage"
source_asset: dike
source_location: Fischbeck
impact_origin: environmental_impact
impact_type: flooded/inundated
amount: around 150 square kilometres
unit: square kilometres


TEXT: Debris of the destroyed bridges in the Ahr valley was washed into the river.
CI_DAMAGES: infrastructure_type="bridges", location="Ahr valley", damage="structural_damage"
source_asset: bridges
source_location: Ahr valley
impact_origin: environmental_impact
impact_type: waste fills and debris
amount: None
unit: None


Now extract all environmental impacts for each infrastructure case.
TEXT: {text}
CI_DAMAGES: {ci_damages}
source_asset: <your_response>
source_location: <your_response>
impact_origin: <your_response>
impact_type: <your_response>
amount: <your_response>
unit: <your_response>

""",
        ),
        (
            "human",
            """
TEXT: {text}
CI_DAMAGES: {ci_damages}
""",
        ),
    ]
)

# location rules: the same "IMPORTANT" points about locations as in the prompt of LLM-extractor (asset_information_prompt)
location_rules = [
    line.strip()
    for line in asset_information_prompt.messages[0].prompt.template.split("Example Input")[0].split("\n")
    if line.strip().startswith("-") and "location" in line.lower() and "damage" not in line.lower()
]
assert location_rules, "no location rules found in asset_information_prompt"

# additional rules per field, on top of the field definition of the schema
JUDGE_EXTRA_RULES = {
    "infrastructure_type": [
        "Use the INFRASTRUCTURE ENTITIES DETECTED BY NER as guidance and keep their phrasing, as LLM-extractor had to.",
        "Adapt it only if the TEXT does not support it; then use the phrasing of the matching NER entity or of the TEXT. Never invent infrastructure.",
        "Do not add articles, quantifiers or numbers (e.g. 'the', 'several', 'two') from the NER entities; numbers belong to infrastructure_amount.",
    ],
    "infrastructure_amount": [
        "Number of affected assets of this infrastructure_type explicitly reported in the TEXT. Preserve the original wording (e.g. 'more than 68'). Do not calculate or infer values. Use null if no number is stated.",
    ],
    "infrastructure_name": [
        "Proper name or designation of this infrastructure_type explicitly mentioned in the TEXT (e.g. 'A1', 'Gates Avenue', 'Braunschweig chemical plant'). Preserve the original wording. Do not use generic infrastructure types as names. Use null if no name is stated.",
    ],
    "location": [
        "Re-check the location and adapt it, if needed, to the most detailed location the TEXT states for this infrastructure, following these rules:",
        *location_rules,
    ],
    "damage": [
        "Use the damage category whose definition matches the damage the TEXT reports for this infrastructure. If several categories apply, use the most severe one (the first one in the definition).",
        "Use null if no damage or disruption is explicitly stated for this infrastructure.",
    ],
    "economic_impact_origin": [
        "Only use the allowed keywords (several keywords comma-separated if the record lists several economic impacts).",
    ],
    "societal_impact_origin": [
        "Only use the allowed keywords (several keywords comma-separated if the record lists several societal impacts).",
    ],
    "economic_impact_amount": ["Preserve the original wording of the TEXT. Do not calculate or infer values."],
    "societal_impact_amount": ["Preserve the original wording of the TEXT. Do not calculate or infer values."],
    "environmental_impact_origin": [
        "Only use the allowed keywords (several keywords comma-separated if the record lists several environmental impacts).",
    ],
    "environmental_impact_amount": ["Preserve the original wording of the TEXT. Do not calculate or infer values."],
    "secondary_infrastructure_asset": [
        "Must be a different infrastructure than infrastructure_type, affected as a consequence of the damage of infrastructure_type.",
        "Use the INFRASTRUCTURE ENTITIES DETECTED BY NER as guidance and keep their phrasing, as LLM-extractor had to. Never invent infrastructure.",
    ],
}


judge_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict fact-checker (LLM-as-a-judge) for information about impacts of natural hazards on critical infrastructure.

You get a TEXT, the INFRASTRUCTURE ENTITIES DETECTED BY NER in this TEXT, and a list of RECORDS that another model (LLM-extractor) extracted from this TEXT.
Check each record against the TEXT and the FIELD RULES below. LLM-extractor had to follow the same FIELD RULES.

Rules:
- Use ONLY the TEXT. Do not use background knowledge, do not infer, do not guess, and do not add information that is not explicitly stated in the TEXT.
- Judge EVERY record. Never remove, merge, split or add records. Return exactly one judgement per record_id.
- Choose one verdict per record:
  - "correct": all fields of the record are supported by the TEXT and follow the FIELD RULES. A field that is null is fine when the TEXT does not state this information.
  - "adapted": the TEXT states that the infrastructure (infrastructure_type) of the record is affected, but one or more fields are wrong, incomplete, missing,
    or do not follow the FIELD RULES (e.g. a location that is less detailed than stated in the TEXT, a damage that is not one of the allowed keywords).
    Correct ONLY these fields and give them in "adapted_fields". Use the wording of the TEXT (same spelling). Set a field to null if its value is not stated in the TEXT.
    A wrong location, damage or amount is a reason to adapt the record, NOT to mark it as false.
  - "false": the TEXT does not mention the infrastructure of the record, or does not state that it is affected. Do not adapt false records.
- In "adapted_fields" only use field names that exist in the record. Never change "record_id".
- "evidence": copy the sentence(s) from the TEXT verbatim that support your verdict. Use an empty string if nothing in the TEXT supports the record.
- "reason": one short sentence explaining your verdict and, for "adapted", what you changed.

FIELD RULES:
{field_rules}

Return ONLY a JSON list, without any other text:
[{{"record_id": 0, "verdict": "correct" | "adapted" | "false", "adapted_fields": {{"<field>": "<corrected value or null>"}}, "evidence": "...", "reason": "..."}}]
""",
        ),
        (
            "human",
            "TEXT:\n{text}\n\nINFRASTRUCTURE ENTITIES DETECTED BY NER: {ner_entities}\n\nRECORDS:\n{records}",
        ),
    ]
)

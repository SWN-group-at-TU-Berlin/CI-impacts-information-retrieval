# Extracted features

The pipeline writes one file `llm_chain_judge_<citation>.csv` per document to `PATH_RESULTS`. Each row is one **infrastructure case**: an infrastructure asset or service damaged by a natural hazard (`infrastructure_type`, `damage`, `location`, ...).
The cascading impacts of this asset (economic, societal, environmental and secondary infrastructure impacts) are extracted separately by LLM-extractor and attached to the row of their source asset.
If an asset has several impacts of the same kind, the values are joined in one cell, separated by `, `.

The tables below list the extracted columns, their description (the same text the LLMs get as field definition), their type and, for categorical columns, the allowed values.

<!-- BEGIN GENERATED (docs/generate_extracted_features.py), do not edit by hand -->

### Infrastructure case

Schema: `InfrastructureCase` in [`src/models/schemas.py`](../src/models/schemas.py)

| Column | Description | Type | Allowed values |
|---|---|---|---|
| `infrastructure_type` | Infrastructure assets or services damaged by a natural hazard. | free text | – |
| `infrastructure_amount` | Number of affected infrastructure assets or services (infrastructure_type) explicitly reported in the text, if any. Preserve the original wording. Example: In the villages around Berlin around 100 roads were damaged. infrastructure_type:'roads', infrastructure_amount:'around 100' | free text (original wording, not converted to a number) | – |
| `infrastructure_name` | Proper name or designation of the affected infrastructure asset (infrastructure_type) explicitly mentioned in the text, if any. Preserve the original wording. Examples: 'A1', 'B510', 'Gates Avenue', 'Braunschweig chemical plant', 'Berlusconi Airport'. Example: The A1 highway was flooded near Cologne. infrastructure_type:'A1 highway', infrastructure_name:'A1' | free text | – |
| `location` | Location explicitly associated with this infrastructure asset. | free text | – |
| `damage` | General category of the damage, failure, disruption, or operational status explicitly associated with this infrastructure asset. | categorical | see [damage categories](#damage-categories) |

### Economic impacts

Schema: `EconomicImpact` in [`src/models/schemas.py`](../src/models/schemas.py)

| Column | Description | Type | Allowed values |
|---|---|---|---|
| `economic_impact_origin` | Whether the impact is a direct cost of repairing, restoring, replacing or recovering damaged infrastructure, or a downstream economic consequence caused by infrastructure damage or disruption. | categorical | `repair_recovery_cost`, `downstream_economic_consequence`, `other_costs_or_losses` |
| `economic_impact_type` | Specific type of economic impact, for example: affected employees, affected companies, affected customers, repair cost, reconstruction/replacement cost, total/general cost, total/general losses, business interruption, supply-chain disruption, production interruption or loss, freight or logistics or trade disruption, revenue loss, agricultural loss, insurance loss, additional operating cost, or other explicitly reported economic consequence. | free text | – |
| `economic_impact_amount` | Quantitative costs explicitly reported in the text, if any. Preserve the original wording and do not calculate or infer values. Examples: '€5 million', 'more than £20,000', 'approximately 10 000 Euros'. | free text (original wording, not converted to a number) | – |
| `economic_impact_unit` | Currency or unit associated with amount, if explicitly stated. Examples: 'EUR', '€', 'euros', 'GBP' | free text | – |

### Societal impacts

Schema: `SocietalImpact` in [`src/models/schemas.py`](../src/models/schemas.py)

| Column | Description | Type | Allowed values |
|---|---|---|---|
| `societal_impact_origin` | Category of the societal impact: 'evacuation_or_displacement' (people or households evacuated, displaced, relocated or made homeless); 'loss_of_critical_service_or_inaccessibility' (people, households or communities without electricity, water, gas, heating or telecommunication, without access to healthcare, schools or other essential services or facilities, or communities isolated or cut off); 'disruption_of_mobility' (people, passengers or commuters unable to travel, stranded or with disrupted journeys); 'other_affected_population' (any other explicitly reported consequence for people, households or communities, e.g. generally affected people or disrupted living conditions, missing people, fatalities). | categorical | `evacuation_or_displacement`, `loss_of_critical_service_or_inaccessibility`, `disruption_of_mobility`, `other_affected_population` |
| `societal_impact_type` | Specific type of societal impact, for example: affected households / families / people / passengers / commuters / students or pupils / population /community / residential area, who are displaced, affected by loss or access to essential services or by disruption of mobility. | categorical (mapped from free text) | `households`, `students`, `passengers`, `community`, `population`, `people` |
| `societal_impact_amount` | Quantitative magnitude explicitly stated in the text, if available. Examples: 'up to 300', 'approximately 10,000', '1.000', 'around 10 000'. Do not calculate or infer values. | free text (original wording, not converted to a number) | – |
| `societal_impact_unit` | Unit associated with the quantitative amount, if explicitly stated. Examples: 'households', 'people', 'families', 'passengers', 'patients', 'students', 'residents', 'citizens', 'commuters', 'farmers', 'communities', 'municipalities'. | free text | – |

### Environmental impacts

Schema: `EnvironmentalImpact` in [`src/models/schemas.py`](../src/models/schemas.py)

| Column | Description | Type | Allowed values |
|---|---|---|---|
| `environmental_impact_origin` | Category of the impact: 'environmental_impact' (any explicitly reported consequence for the natural environment, e.g. land, soil, groundwater, rivers or regions). | categorical | `environmental_impact` |
| `environmental_impact_type` | Specific type of environmental impact, for example: 'waste fills and debris' (waste, debris or rubble deposited on land, in or along rivers); 'inundated' (flooded or inundated land, fields, areas, or regions); 'pollution of environment' (land, soil, groundwater, rivers etc.) | free text | – |
| `environmental_impact_amount` | Quantitative magnitude explicitly stated in the text, if available. Preserve the original wording and do not calculate or infer values. Examples: '15 cm thick', 'around 100 hectares', 'more than 2,000 tonnes', '7 out of 31'. | free text (original wording, not converted to a number) | – |
| `environmental_impact_unit` | Unit associated with the quantitative amount, if explicitly stated. Examples: 'cm', 'hectares', 'square kilometres', 'km²', 'tonnes', 'litres', 'cubic metres'. | free text | – |

### Secondary (cascading) infrastructure impacts

Schema: `SecondaryInfrastructureImpact` in [`src/models/schemas.py`](../src/models/schemas.py)

| Column | Description | Type | Allowed values |
|---|---|---|---|
| `secondary_infrastructure_asset` | Other infrastructure asset or infrastructure-related service affected as a consequence of the source damage. | free text | – |

### Damage categories

Allowed values of `damage`, ordered from most to least severe. Each infrastructure case gets ONE category: if the text mentions several damages, the one that refers directly to the `infrastructure_type` is used, and only if several refer equally well to it, the most severe one. Free-text damages of the LLM (e.g. 'severely damaged') are mapped to a category via the keywords, see `DAMAGE_CATEGORIES` in [`src/postprocess.py`](../src/postprocess.py).

| Category | Definition | Keywords mapped to this category |
|---|---|---|
| `structural_damage` | physical damage to or destruction of the asset itself | damaged, severely damaged, highly damaged, destroyed, largely destroyed, heavily destroyed, dam failure(s), dyke break(s), dike break(s), breached dike(s), broken embankment(s), derailed |
| `inundation` | the asset is flooded or covered by water | flooded, inundated, water on |
| `contamination` | the asset or its service (e.g. drinking water) is contaminated or polluted | contaminated, polluted |
| `service_loss` | the service (only water/wastewater/telecommunication/IT/energy/gas) provided by the asset is lost or the assets' capacity is limited or exceeded, e.g. outage, blackout, overloaded/overwhelmed, congested network(s), supply of water, electricity or gas | outage(s), blackout, loss of water/electricity/gas/telecommunication/it, no supply, limited/exceeded capacity, overwhelmed, overheated, left without water/electricity/gas/telecommunication/it, failure(s) |
| `obstruction` | the asset is obstructed by debris or trees | obstruction, debris, debris on, fallen tree(s), broken tree(s), overturned tree(s), uprooted tree(s), felled tree(s) |
| `disruption` | the operation of the asset (only transportation) is disrupted, restricted, or stopped, e.g. closures, blockages, delays, cancellations, traffic jams | disrupted, blocked, impassable, partly closed, closed, closure(s), (partly) out of service, cut off, traffic jam(s), traffic disruption(s), traffic congestion(s), delay(s), flight delay(s), train delay(s), train failure(s), cancellation(s), canceled |
| `affected` | the asset is reported as (indirectly) affected or impacted without further details, or reported as (indirectly) affected/impacted by a hazard (e.g., fire, wind, flood, heat, drought) without further details | affected, affected by a hazard, indirectly affected |
| `restored_recovered` | the asset or its service is restored, repaired, rebuilt or reconnected | re-established, reestablished, restored, reconnected, rebuild, repaired, debris removal |

### Infrastructure groups

Allowed values of `infrastructure_group` (added in postprocessing): `infrastructure_type` is matched against the CI patterns in [`ner_patterns.jsonl/patterns`](../ner_patterns.jsonl/patterns) and gets the `subgroup_name` of the matching pattern. Records whose type matches no pattern (e.g. theatres, stadiums) are removed.

<details><summary>Show all 64 groups</summary>

| `infrastructure_group` | Group | Sector |
|---|---|---|
| `education_general` | education | Education |
| `education_kita` | education | Education |
| `education_school` | education | Education |
| `electricity_distribution` | electricity | Energy |
| `electricity_general` | electricity | Energy |
| `electricity_generation` | electricity | Energy |
| `electricity_others` | electricity | Energy |
| `electricity_supply` | electricity | Energy |
| `electricity_transmission` | electricity | Energy |
| `electricity_renewable_hydro` | electricity_renewable | Energy |
| `electricity_renewable_offshore` | electricity_renewable | Energy |
| `electricity_renewable_solar` | electricity_renewable | Energy |
| `electricity_renewable_thermal` | electricity_renewable | Energy |
| `electricity_renewable_wind` | electricity_renewable | Energy |
| `energy_general` | energy | Energy |
| `energy_others` | energy | Energy |
| `energy_supply` | energy | Energy |
| `gas_distribution` | gas | Energy |
| `gas_lng` | gas | Energy |
| `gas_storage` | gas | Energy |
| `gas_supply` | gas | Energy |
| `hydrogen_general` | hydrogen | Energy |
| `oil_others` | oil | Energy |
| `cables_general` | cables | Energy/Communication |
| `cables_poles` | cables | Energy/Communication |
| `energy_or_water_storage` | energy_or_water | Energy/water |
| `healthcare_general` | healthcare | Health |
| `healthcare_hospitals` | healthcare | Health |
| `healthcare_others` | healthcare | Health |
| `healthcare_services` | healthcare | Health |
| `nursing_general` | nursing | Health |
| `nursing_homes` | nursing | Health |
| `industry_chemical` | industry | Heavy Industry |
| `industry_heavy` | industry | Heavy Industry |
| `oil_refinement` | oil | Heavy Industry |
| `digital_communication_services` | digital | IT/Communication |
| `digital_data_centres` | digital | IT/Communication |
| `digital_general` | digital | IT/Communication |
| `digital_networks` | digital | IT/Communication |
| `flood_protection_barriers` | flood_protection | Structural Protection Measures |
| `flood_protection_general` | flood_protection | Structural Protection Measures |
| `aviation_airports` | aviation | Transportation |
| `aviation_general` | aviation | Transportation |
| `aviation_transport` | aviation | Transportation |
| `rail_general` | rail | Transportation |
| `rail_infrastructure` | rail | Transportation |
| `rail_metro_infrastructure` | rail | Transportation |
| `rail_metro_transport` | rail | Transportation |
| `rail_transport` | rail | Transportation |
| `road_network` | road | Transportation |
| `shipping_general` | shipping | Transportation |
| `shipping_inland_waterways` | shipping | Transportation |
| `shipping_ports` | shipping | Transportation |
| `shipping_services` | shipping | Transportation |
| `transport_general` | transport | Transportation |
| `transport_others` | transport | Transportation |
| `waste_general` | waste | Waste |
| `waste_treatment` | waste | Waste |
| `wastewater_collection` | wastewater | Wastewater |
| `wastewater_general` | wastewater | Wastewater |
| `wastewater_treatment` | wastewater | Wastewater |
| `drinking_water_distribution` | drinking_water | Water |
| `drinking_water_general` | drinking_water | Water |
| `drinking_water_supply` | drinking_water | Water |

</details>

<!-- END GENERATED -->

## Further columns

| Column | Added by | Description |
|---|---|---|
| `citation_id` | pipeline | citation taken from the file name, e.g. `Koks 2022` |
| `chunk_id` | pipeline | number of the text chunk within the document |
| `chunk_text` | pipeline | text of the chunk the record was extracted from |
| `is_infrastructure_case` | LLM-extractor | always `True`, as only chunks that describe damages to critical infrastructure caused by a natural hazard are processed further in the pipeline. All chunks with `False` are skipped by the LLM-extractor |
| `has_cascading_impacts` | LLM-extractor | `True` if the chunk describes cascading impacts of the damaged infrastructure |
| `assets_org` | LLM-extractor | infrastructure cases as first extracted, kept to trace changes made in the cascading-impact step |
| `infrastructure_group` | postprocessing | CI group of `infrastructure_type`, see [infrastructure groups](#infrastructure-groups) |
| `location_orig` | postprocessing | `location` before cleaning (e.g. removing "the", "between") |
| `judge_verdict` | LLM-as-a-judge | `correct`, `adapted`, `false` or `not_judged` (no valid verdict, record kept unchanged) |
| `judge_adapted_fields` | LLM-as-a-judge | JSON `{field: {"old": ..., "new": ...}}` of the fields changed by the judge (`adapted` records) |
| `judge_reason` | LLM-as-a-judge | short explanation of the verdict |
| `judge_evidence` | LLM-as-a-judge | sentence(s) of the chunk text the verdict is based on |
| `judge_evidence_in_text` | LLM-as-a-judge | `True` if the evidence is found verbatim in the chunk text |
| `judge_adaptation_in_text` | LLM-as-a-judge | `True` if all adapted values are found in the chunk text, empty if nothing was adapted |

## Updating this page

The section between the `BEGIN GENERATED` / `END GENERATED` markers is created from [`src/models/schemas.py`](../src/models/schemas.py), [`src/postprocess.py`](../src/postprocess.py) and [`ner_patterns.jsonl/patterns`](../ner_patterns.jsonl/patterns). After changing one of these files, run

```bash
uv run python docs/generate_extracted_features.py
```

The pre-commit hook `extracted-features-doc` runs it automatically when one of these files is committed.

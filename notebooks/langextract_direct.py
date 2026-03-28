# %% [markdown]
# 
# # Aim:
# How to link the rule-based extracted CI_TYPE-GEO pairs with the respective Ci failure impacts
# 
# **Idea**\
# Test using prompt engineering by passing table of CIGEO pairs to GPT-J model.
# Steps:
# * Load model and apply it always on one chunk of the document to extract CI failure impacts 
# * Use prompt engineering to extract time and location of the CI failure (origin) the CI impacts (impact location)
# or 
# * Pass dataframe of pairs as input to the model
# or
# * Use few shot prompting with example answers
# 
# **Finally:**
# * Compaire all approaches of spatial and temporal linking CI failure impacts
# 
# 

# %%
# %env CUDA_DEVICE_ORDER=PCI_BUS_ID
# %env CUDA_VISIBLE_DEVICES=0  # nvidia gpu
# %env PYTORCH_ALLOC_CONF=expandable_segments:True
# # %env TORCH_CUDA_ARCH_LIST=8.6

# # settings for distributed computing
# %env WORLD_SIZE=1
# %env RANK=0
# %env LOCAL_RANK=0

# # NOTE: # WORLD_SIZE: each GPU corresponds to one process (world = no. of processes within a group), processes communicate with each other enabling eg., distributed training
# # NOTE: # RANK: IDs of the processes, ranging from 0 up to WORLD_SIZE - 1

# %%
import os
import sys
import argparse
import re
from datetime import datetime
import subprocess
from glob import glob
from pathlib import Path
from itertools import chain

import pandas as pd 
import numpy as np
import spacy
import textwrap
print("Loading packages for lx...")
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_docling import DoclingLoader
from huggingface_hub import login
import langextract as lx

# need for Langchain-Docling
pipeline_options = PdfPipelineOptions()
pipeline_options.allow_external_plugins = True 

print("Loading settings")
sys.path.append("./")
from src.settings import settings as s
import src.document_cleaning as dc


# set default location to store model before loading transformers
os.environ["HF_HOME"] = s.HF_HOME_DIR

login(token=os.environ.get("HF_TOKEN"))

test_mode = False


# %%
print("Loading user args")
MODEL_CHOICE = "llama3"  # default ollama model

try:
    # load user arguments Ollama server and model
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host_port",
        type=str,
        default="11434",
        help="The host and port of the llama server",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_CHOICE,
        help="The name of the llama model to use",
    )
    args = parser.parse_args()

    host_port = args.host_port
    model_name = args.model_name

except:
    host_port = "11434"
    model_name = MODEL_CHOICE
    
print("Using host and port:", host_port)
print("Using model:", model_name)

# %%
## NOTE: make sure to set project root as working dir

# Input paths
PARSED_TEXT_DIR = "./" + s.PATH_DATA + "parsed_documents/"

# CI GEO pairs
filename_ci_geo_entities = "extracted_ci_geo_entities.csv"
CI_GEO_FILEPATH = Path("./"  + s.PATH_DATA) / filename_ci_geo_entities
NER_PATTERNS_FILEPATH = s.NER_PATTERNS_FILEPATH

## LX outputs
LX_OUTPUTS_DIR = s.PATH_LX_DATA

## document to process
docs_list = glob(PARSED_TEXT_DIR + "*_cleaned.md")
lx_response_filename = s.LX_DATA_FILENAME.replace(".csv", ".jsonl")
lx_response_filename_df = s.LX_DATA_FILENAME
LX_OUTPUTS_FILEPATH = Path(LX_OUTPUTS_DIR, lx_response_filename)
LX_OUTPUTS_DF_FILEPATH = Path(
    LX_OUTPUTS_DIR, lx_response_filename_df
)  # TODO make as part of OUTPUT_LX_FILEPATH and replaced suffix


if test_mode:
    print("Test mode is ON. Using only a small sample of documents for testing.")
    docs_list_sample = [
        # Path(PARSED_TEXT_DIR, "Keller 2014 - Mapping Natural Hazard Impacts on Road Infrastructure—The Extreme Precipitation in Baden-Württemberg_cleaned.md"),
        Path(PARSED_TEXT_DIR,"Rozendaal 2021 - Infrabel_ Flood damage to railway track worth tens of millions of euros _ SpoorPro - incomplete_cleaned.md",),
        Path(PARSED_TEXT_DIR,"The Guardian 2018 - Freezing weather costs UK economy £1bn a day _ UK weather_cleaned.md",),
        Path(PARSED_TEXT_DIR, "Brown 2010 - Economy feels chill as UK grinds to a halt _ The Independent_cleaned.md"),
        #Path(PARSED_TEXT_DIR, "Artemis 2015 - PERILS finalises Storm Desmond UK flood loss estimate at £604m_cleaned.md"),
        #Path(PARSED_TEXT_DIR, "Treanor 2015 - Storm Desmond damage across Cumbria estimated at £500m _ Storm Desmond _The Guardian_cleaned.md"),
        #Path(PARSED_TEXT_DIR, "PWC 2015 - Updated estimates on cost of Storm Desmond_cleaned.md"),
    ]


# %% [markdown]
# # Generate CI_GEO pairs

# %%

# %%
print("Try loading spaCy language model for remote instance (e.g., cluster)")
try:
    nlp = spacy.load(s.SPACY_MODEL)
except (OSError, ValueError):
    print(f"spaCy language model '{s.SPACY_MODEL}' not found. Downloading ...")
    subprocess.check_call(["uv", "pip", "install", "spacy-transformers"])
    subprocess.check_call(
        ["uv", "run", "python", "-m", "spacy", "download", s.SPACY_MODEL]
    )
    nlp = spacy.load(s.SPACY_MODEL)


# %%
## see for more info: https://spacy.io/usage/rule-based-matching#entityruler
## NOTE EntityRuler is hidden inside .add_pipe()


## call nlp model and create pipeline with new entity pattern
# NOTE Creation of the new entity (CI_TYPE) solves the issue that FAC entities (buildings, airports, highways, bridges, etc.) refer only to the name of the facility (e.g. A76, Ahrtalbahn)
config = {"spans_key": None, "annotate_ents": True, "overwrite": False}
try:
    ruler = nlp.add_pipe("span_ruler", config=config)
    ruler.from_disk(NER_PATTERNS_FILEPATH)
except ValueError:
    print("SpanRuler already exists in pipeline.")
    ruler = nlp.get_pipe("span_ruler")
    ruler.from_disk(NER_PATTERNS_FILEPATH)


# %%


# %%


## load docs
docs_list = glob(PARSED_TEXT_DIR + "*_cleaned.md")
print(f"Found {len(docs_list)} cleaned documents.")


if test_mode:
    docs_list = docs_list_sample

## DataFrame to store CI-GEO entity pairs
df_ci_geo = pd.DataFrame(  ## TODO make as pydantic class with fixed attributes
    columns=[
        "document_id",
        "chunk_id",
        "ci_entity",
        "ci_entity_label",
        "geo_entity",
        "geo_entity_label",
        "token_distance",
    ]
)


## iterate over all cleaned documents and extract CI-GEO entity pairs
# but first check if output file already exists
if CI_GEO_FILEPATH.exists():
    print(
        f"\nCI-Geo entities already exists, see file:  {CI_GEO_FILEPATH}. \nLoading file from disk"
    )

    with open(CI_GEO_FILEPATH, "r") as f:
        df_ci_geo = pd.read_csv(f)

else:
    for FILE_PATH in docs_list:
        print(f"\n\n------- Loading  - {Path(FILE_PATH).name} -------- ")
        loader = DoclingLoader(FILE_PATH)  # use chunks from Docling.Loader
        doc = loader.load()

        ## get most likely geolocation for each CI entity based on distance
        for i, chunk in enumerate(doc):
            nlp_chunk = nlp(chunk.page_content)
            all_ents = [ent for ent in nlp_chunk.ents]
            ci_type_ents = [
                ent for ent in nlp_chunk.ents if ent.label_ in ["CI_TYPE", "FAC"]
            ]
            ci_type_ents_info = [
                ent for ent in nlp_chunk.ents if ent.label_ in ["CI_TYPE"]
            ]
            fac_ents_info = [ent for ent in nlp_chunk.ents if ent.label_ in ["FAC"]]

            # check if chunk contains CI_TYPE entities
            if len(ci_type_ents) > 0:
                print(
                    f"\nChunk [{i}], No. CI_TYPE and FAC entities: {len(ci_type_ents)}"
                )
                print(
                    f"Contains following entities for CI_TYPE: {ci_type_ents_info}, FAC: {fac_ents_info}"
                )
                print(f"Chunk text [{i}]:", chunk.page_content)
                # print(f"{ {(ci_type_ents[i].text, ci_type_ents[i].label_) for i in range(len(ci_type_ents))} } ")

                # iterate over all entities within chunk
                for ent_idx in range(len(all_ents)):
                    # when entity is CI_TYPE or FAC (i.e. buidling, airports, highways) do following ...
                    if all_ents[ent_idx].label_ in ["CI_TYPE", "FAC"]:
                        ci_idx = ent_idx

                        ## .. calculate distances between CI_TYPE entity and  all GEO entities in chunk based on index position
                        distance_list = []
                        idx_in_chunk = []
                        try:
                            for ent_idx in range(len(all_ents)):
                                # TODO calc distances between CI_TYPE ~ GEO entities based on word numbers and not entities
                                if all_ents[ent_idx].label_ in ["GPE", "LOC"]:
                                    geo_idx = ent_idx
                                    dist_ent_pair = np.abs(ci_idx - geo_idx)
                                    distance_list.append(dist_ent_pair)
                                    idx_in_chunk.append((ent_idx))
                                    closest_pair_idx = np.argmin(
                                        distance_list
                                    )  # idx of closest GEO entity
                                    distance_closest_pair = distance_list[
                                        closest_pair_idx
                                    ]

                            threshold = (
                                5  # max token distance between CI_TYPE and GEO entity
                            )
                            if distance_closest_pair > threshold:
                                print(
                                    f""" Token distance between CI_TYPE/FAR and next GEO entity is {distance_closest_pair} and thus larger than the allowed distance of {threshold} tokens """
                                )
                                continue
                            else:
                                print(
                                    f"""  Closest GEO entity to CI_TYPE/FAC entity "{all_ents[ci_idx]}" is "{all_ents[idx_in_chunk[closest_pair_idx]]}" at distance {distance_closest_pair}"""
                                )  # TODO constrain min.distance to max value (eg. 5 tokens), issue: likely when distance value is high that geolocation of Ci_type is mentioned in previous sentences or chunk

                            ## write as dict entry incl chunk_id, ci_entity, geo_entity, distance
                            result_dict = {
                                "document_id": Path(FILE_PATH).stem,
                                "chunk_id": i,
                                "ci_entity": all_ents[ci_idx].text,
                                "ci_entity_label": all_ents[ci_idx].label_,
                                "geo_entity": all_ents[
                                    idx_in_chunk[closest_pair_idx]
                                ].text,
                                "geo_entity_label": all_ents[
                                    idx_in_chunk[closest_pair_idx]
                                ].label_,
                                "token_distance": distance_closest_pair,
                            }
                            df_ci_geo = pd.concat(
                                [df_ci_geo, pd.DataFrame([result_dict])],
                                ignore_index=True,
                            )

                        except IndexError:
                            print("No GEO entities found in this chunk.")
                            continue
                        # print("\nidx_in_chunk, closest pair idx", idx_in_chunk, closest_pair_idx)

                        # spacy.displacy.render(
                        #     nlp_chunk, style="ent",
                        #     options={"ents": ["CI_TYPE", "GPE", "LOC", "FAC"], "colors": {"CI_TYPE": "violet"}}
                        # )

                else:
                    print("\nNo CI_TYPE or FAC entities found in this chunk.")
                    continue

    # save to disk when not existing
    with (CI_GEO_FILEPATH).open("w") as f:
        df_ci_geo.to_csv(f, index=False)
    print(f"\nSaved extracted CI-GEO entity pairs to {CI_GEO_FILEPATH}")

# %%


# %% [markdown]
# ## LLama with LangExtract
# 

# %% [markdown]
# ## Prompt text

# %%
# Prompt and extraction rules

prompt_with_cigeo = textwrap.dedent(
    """
    Extract information from the context about the affected infrastructure_type, its damage, its geolocation.

    Use the exact text for extractions. DO NOT paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.


    Provide in the field "infrastructure_type" the type of infrastructure that was affected.
    If no information about the infrastructure type is found, then return for this field a "NAN" value.

    Provide in the field "geolocation" the location of the affected infrastructure.
    If no information about the location of the affected infrastructure is found, then return for this field a "NAN" value.

    Provide in the field "damage" the type of damage of the affected infrastructure.
    Use for example phrases like [partly closed, outages, largely destroyed, heavily destroyed, contaminated, out of service, severely damaged, largely destroyed, due to water/debris/risk aquaplaning on road, dam failure, closures, derailed, impassable, debris, destroyed, indirectly affected (through disrupted road and rail traffic), disrupted, blocked (through trees, roofs), damaged, affected, little to no]
    If no information about the damage type is found, then return for this field a "NAN" value.

    Provide in the field "name" the name of the affected infrastructure facility, such as a name of an airport or the highway number (e.g. A-3, A5, AP-7).
    If no information about the name of the affected infrastructure is found, then return for this field a "NAN" value.


    Finally, evaluate and improve your answer.
    In particular, for the fields ""infrastructure_type" and "geolocation" you should evaluate and improve your answer based on the information mentioned in CI locations.
    
    CI locations:
    {% for item in context %}
    - ("ci_entity" and "geo_entity":\n {{ item["ci_locations"][["ci_entity", "geo_entity"]] }})
    {% endfor %}




    """
)


# %%

prompt_no_cigeo = textwrap.dedent(
    """
    Extract information from the context about the affected infrastructure_type, its damage, its geolocation.

    Use the exact text for extractions. DO NOT paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.


    Provide in the field "infrastructure_type" the type of infrastructure that was affected.
    If no information about the infrastructure type is found, then return for this field a "NAN" value.

    Provide in the field "geolocation" the location of the affected infrastructure.
    If no information about the location of the affected infrastructure is found, then return for this field a "NAN" value.

    Provide in the field "damage" the type of damage of the affected infrastructure.
    Use for example phrases like [partly closed, outages, largely destroyed, heavily destroyed, contaminated, out of service, severely damaged, largely destroyed, due to water/debris/risk aquaplaning on road, dam failure, closures, derailed, impassable, debris, destroyed, indirectly affected (through disrupted road and rail traffic), disrupted, blocked (through trees, roofs), damaged, affected, little to no]
    If no information about the damage type is found, then return for this field a "NAN" value.

    Provide in the field "name" the name of the affected infrastructure facility, such as a name of an airport or the highway number (e.g. A-3, A5, AP-7).
    If no information about the name of the affected infrastructure is found, then return for this field a "NAN" value.

    Finally, evaluate and improve your answer.


    """
)


# %%
## Few shot examples


few_shot_examples = [
    ## Skounding 2023
    lx.data.ExampleData(
        text="Elsewhere, the Mediterranean country has been battered by severe storms. On overnight storm in Milan on Monday tore off roofs and uprooted trees, blocking roads and disrupting overground transportation in Italy’s financial capital.",
        extractions=[
            lx.data.Extraction(
                extraction_class="infrastructure_type",
                extraction_text="roads",
                attributes={"damage": "blocked", "geolocation": "Milan (city)"},
            ),
        ],
    ),
    ## Ferlita 2023
    lx.data.ExampleData(
        text="Another fire broke out in Pioppo, a hamlet of Monreale in the Palermo area, where a fire threatened several homes in Casaboli and destroyed much of the vegetation in that area."
        "The Partinico area was also not spared: the fire broke out a few days ago on State highway 113.",
        extractions=[
            lx.data.Extraction(
                extraction_class="infrastructure_type",
                extraction_text="highway",
                attributes={
                    "damage": "affected",
                    "geolocation": "Partinico area",
                    "name": "State highway 113",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Further damage occurred in recent days at Catania airport, considered the fifth most important in Italy. Following a fire that broke out inside the terminals, the access areas were promptly closed to travelers, causing enormous damage to the local economy, the tourism sector, and various professionals. In monetary terms, the damage is enormous: the National Civil Aviation Authority estimates a total investment for facilities and maintenance of around €200,000. The MEC (Consumer Voters Movement), on the other hand, estimates a cost of around €40 million per day due to the closure of the airport as a result of the fire. The total damage is estimated at more than €80 million. The fire broke out last Sunday night at Vincenzo Bellini Airport in Catania and, after two days of relentless work to extinguish the fire, arrivals and departures resumed from Terminal C.",
        extractions=[
            lx.data.Extraction(
                extraction_class="infrastructure_type",
                extraction_text="terminals",
                attributes={
                    "damage": "damaged",
                    "geolocation": "Catania",
                    "name": "Vincenzo Bellini Airport",
                },
            ),
            #         lx.data.Extraction(
            #             extraction_class="impacts_to_other_infrastructure_assets",
            #             extraction_text="airport",
            #             attributes={"damage": "closure", "geolocation": "Catania"
            #             },
            #         ),
            #         lx.data.Extraction(
            #             extraction_class="economic_impact",
            #             extraction_text="€200,000",
            #             attributes={
            #                 "damage": "total investment costs for facilities and maintenance",
            #                 "geolocation": "Catania Airport"
            #             },
            #         ),
            #         lx.data.Extraction(
            #             extraction_class="economic_impact",
            #             extraction_text="€80 million",
            #             attributes={
            #                 "damage": "total damage",
            #                 "geolocation": "Catania Airport"
            #             },
            #         )
        ],
    ),
    ## EFE 2024
    lx.data.ExampleData(
        text="The most serious disruptions are on roads in Valencia, with closures on several sections of the A-3, the A-7 and the AP-7, as well as on the roads that connect it with Alicante and the N-3, N-322, N-330 and N-332 roads as they pass through the towns of Picassent, La Alcudia, Requena, Utiel, Buñol, Sueca, Algemesí, Guadassuar, Alzira or Chiva (Valencia), among other municipalities.",
        extractions=[
            lx.data.Extraction(
                extraction_class="infrastructure_type",
                extraction_text="roads",
                attributes={
                    "damage": "closures",
                    "geolocation": "Valencia area",
                    "name": "A-3",
                    # "impacts_to_other_infrastructure_assets": "traffic disrupted"
                },
            ),
            lx.data.Extraction(
                extraction_class="infrastructure_type",
                extraction_text="roads",
                attributes={
                    "damage": "closures",
                    "geolocation": "Valencia area",
                    "name": "A-7",
                    # "impacts_to_other_infrastructure_assets": "traffic disrupted",
                },
            ),
            lx.data.Extraction(
                extraction_class="infrastructure_type",
                extraction_text="roads",
                attributes={
                    "damage": "closures",
                    "geolocation": "Valencia area",
                    "name": "AP-7",
                    # "impacts_to_other_infrastructure_assets": "traffic disrupted"
                },
            ),
            # lx.data.Extraction(
            #     extraction_class="impacts_to_other_infrastructure_assets",
            #     extraction_text="traffic",
            #     attributes={"damage": "disrupted", "geolocation": "Valencia area"},
            # ),
        ],
    ),
    ## Containerlift 2024
    lx.data.ExampleData(
        text="Nevertheless, the reopening of Valencia and Sagunto ports for maritime traffic marks a significant step forward for the region’s recovery.",
        extractions=[
            lx.data.Extraction(
                extraction_class="infrastructure_type",
                extraction_text="ports",  # "Port operability",
                attributes={
                    "damage": "temporarily closed",
                    "geolocation": "Valencia and Sagunto ports",
                },
            ),
            # lx.data.Extraction(
            #     extraction_class="impacts_to_other_infrastructure_assets",
            #     extraction_text="reopening", #" Port operability",
            #     attributes={"damage": "temporarily closed", "geolocation": "Sagunto port"},
            # )
        ],
    ),
    ## Khazai et al 2013
    lx.data.ExampleData(
        text="Durch einen Deichbruch am 10.06. mussten im Landkreis Stendal Fernverkehrsstrecken der Deutschen Bahn AG gesperrt werden, sodass es zu Zugausfällen und langen Verspätungen kommt.",
        extractions=[  # tricky extraction due to non-English language
            lx.data.Extraction(
                extraction_class="infrastructure_type",
                extraction_text="Deichbruch",
                attributes={
                    "damage": "dam failure",
                    "geolocation": "Stendal (Landkreis)",
                },
            ),
            # lx.data.Extraction(
            #     extraction_class="impacts_to_other_infrastructure_assets",
            #     extraction_text="Verspätungen",   # train service
            #     attributes={"damage": "disrupted"},
            # ),
        ],
    ),
    ## Koks et al., 2022
    lx.data.ExampleData(
        text="In Belgium, several towns experienced disruptions in water supply (in particular as a result of pollution).",
        extractions=[
            lx.data.Extraction(
                extraction_class="infrastructure_type",
                extraction_text="water supply",
                attributes={"damage": "polluted", "geolocation": "Belgium"},
            ),
            #         lx.data.Extraction(
            #             extraction_class="impacts_to_other_infrastructure_assets",
            #             extraction_text="water supply",
            #             attributes={"damage": "disrupted"},
            #         ),
        ],
    ),
    # # 1. LAST ONE CHANGED:
    lx.data.ExampleData(
        text="In Belgium, several towns experienced disruptions in water supply (in particular as a result of pollution)."
        "Directly after the event, approximately 3400 families had no access to potable water.",
        extractions=[
            lx.data.Extraction(
                extraction_class="infrastructure_type",
                extraction_text="water supply",
                attributes={"damage": "polluted", "geolocation": "Belgium"},
            ),
            #         # lx.data.Extraction(
            #         #     extraction_class="impacts_to_other_infrastructure_assets",
            #         #     extraction_text="water supply",
            #         #     attributes={"damage": "disrupted"},
            #         # ),
            #         lx.data.Extraction(
            #             extraction_class="societal_impact",
            #             extraction_text="3400 families",
            #             attributes={"damage": "affected"},
            #         ),
        ],
    ),
    lx.data.ExampleData(
        text="Within the region of Rhineland-Palatinate, it took 2 weeks to ensure 100 % coverage again through emergency communication masts."
        "Within 1 month, most of the network was restored to pre-disaster service provision."
        "After 5 months, broadband has also been restored in the most affected areas, which started in most areas only after power infrastructure was rebuilt.",
        extractions=[
            lx.data.Extraction(
                extraction_class="infrastructure_type",
                extraction_text="power infrastructure",
                attributes={
                    "damage": "affected",
                    "geolocation": "Rhineland-Palatinate",
                },  # tricky extraction of location-info
            ),
            # lx.data.Extraction(
            #     extraction_class="impacts_to_other_infrastructure_assets",
            #     extraction_text="broadband",
            #     attributes={"geolocation": "most areas only after power infrastructure was rebuilt"},
            # ),
        ],
    ),
    lx.data.ExampleData(
        text="More than 130 km of motorways were closed directly after the event, of which 50 km were still closed two months later, with an estimated repair cost of EUR100 million (Hauser, 2021). ",
        extractions=[
            lx.data.Extraction(
                extraction_class="infrastructure_type",
                extraction_text="motorways",
                attributes={"damage": "closure"},
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Of the 112 bridges in the flooded 40 km of the Ahr valley (Rhineland-Palatinate), 62 bridges were destroyed, 13 were severely damaged and only 35 were in operation a month after the flood event (MDR, 2021).",
        extractions=[
            lx.data.Extraction(
                extraction_class="infrastructure_type",
                extraction_text="62 bridges",
                attributes={
                    "damage": "destroyed",
                    "geolocation": "Ahr valley",
                    "region": "Rhineland-Palatinate",
                },
            ),
        ],
    ),
]


# %% [markdown]
# ## Apply LangExtract
# 

# %%
## extract CI failure impacts via LangExtract

## load docs
docs_list = glob(PARSED_TEXT_DIR + "*_cleaned.md")

if test_mode:
    docs_list = docs_list_sample


print(f"Found {len(docs_list)} cleaned documents ready for LangExtract.")


responses_all_docs = []
responses_citations = []
responses_chunk_text = []

## iterate over all cleaned documents and extract CI-GEO entity pairs
for i, FILE_PATH in enumerate(docs_list):
    filename_stem = Path(FILE_PATH).stem

    print(
        f"\n\n -------- Processing document [{i + 1}]: {Path(FILE_PATH).name} -------- \n"
    )

    ## extract authors, publication year and title
    citation_pattern = r"(.*?)(\d{4})(.*)"  # split at first occurrence of year
    try:
        authors, year, title = re.findall(citation_pattern, filename_stem)[0]
        citation = f"{authors} {year}"
    except AttributeError as e:
        print(f"Could not extract citation from title: {e}")
        citation = filename_stem

    # ## load doc
    # with open(FILE_PATH, "r") as file:
    #     content = file.read()
    # doc = [lx.data.Document(content)]  # wrap content in Document object
    loader = DoclingLoader(FILE_PATH)  # use chunks from Docling.Loader
    doc = loader.load()

    ## TODO loop makes use of the chunking in docling, however, the implemented chunking strategy in LX might be better,
    ## however, then `[lx.data.Document(content)]` object needs to be splitted into smaller chunks (currently entire text is 1 chunk)

    ## NOTE for multiple documents / long text use batch processing (threading, character buffer, number of model passes on the text)
    responses = []

    # iterate over chunks per document
    for j, chunk_text in enumerate(doc):


        # remove URLs
        chunk_text.page_content = dc.remove_urls(chunk_text.page_content)

        if chunk_text.page_content.strip() == "":
            print(f"----------- Skipping empty chunk id {j} ---------")
            continue

        try:
            df_ci_geo_doc = df_ci_geo.loc[df_ci_geo["document_id"] == filename_stem]
            

            # if Ci-geo pair exists for chunk
            if not df_ci_geo_doc.loc[df_ci_geo_doc["chunk_id"] == j].empty:
                
                df_ci_geo_doc.loc[df_ci_geo_doc["chunk_id"] == j]

                context = [{
                    "prompt": prompt_with_cigeo, 
                    "ci_locations": df_ci_geo_doc.loc[df_ci_geo_doc["chunk_id"] == j],
                },]

                response = lx.extract(
                    text_or_documents=chunk_text.page_content,
                    prompt_description=context, # context (prompt_old) = 3 (1) error-missing exxtraton key; prompt_incl ci_geo=nearly only Errors
                    examples=few_shot_examples,
                    model_id=model_name,
                    model_url=os.getenv("OLLAMA_HOST", f"http://localhost:{host_port}"),
                    # language_model_type=lx.inference.OllamaLanguageModel,
                    temperature=0.2,
                    extraction_passes=1,  # decrease recall -> faster processing
                    # max_workers=4,   # invalid option for ollama
                    max_char_buffer=1024,  # NOTE testing 1042, with 512 (same as for LLM1) no error messages,  not increase due to TImeOutError:       # adapt to max. token sequence length of model
                )

            # Ci-geo pair not exists for chunk
            else:
                response = lx.extract(
                    text_or_documents=chunk_text.page_content,
                    prompt_description=prompt_no_cigeo,
                    examples=few_shot_examples,
                    model_id=model_name,
                    model_url=os.getenv("OLLAMA_HOST", f"http://localhost:{host_port}"),
                    # language_model_type=lx.inference.OllamaLanguageModel,
                    temperature=0.2,
                    extraction_passes=1,  # decrease recall -> faster processing
                    # max_workers=4,   # invalid option for ollama
                    max_char_buffer=1024,  # NOTE testing 1042, with 512 (same as for LLM1) no error messages,  not increase due to TImeOutError:       # adapt to max. token sequence length of model
                )

            responses.append(response)
            responses_citations.append(filename_stem)  # "citation_id" for each response
            responses_chunk_text.append(chunk_text.page_content)  # chunk text to traceback info for each response


        except ValueError as e:
            print("\n------- chunk id", j)
            print(f"Error when applying LangExtract on document {filename_stem}, text block {j}: {e}")
            print("Probably one of the few-shot examples does not match.")
            pass
        except (lx.resolver.ResolverParsingError, lx.core.exceptions.FormatParseError,) as e:
            print("\n------- chunk id", j)
            print(f"Error when applying LangExtract on document {filename_stem}, text block {j}: {e}")
            print("Probably content does not contain an 'extractions' key.")
            print("Respective chunk text:", chunk_text.page_content)
            pass
        except (TimeoutError, lx.core.exceptions.InferenceRuntimeError, AttributeError,) as e:
            print("\n------- chunk id", j)
            print(f"Error when applying LangExtract on document {filename_stem}, text block {j}: {e}")
            print("Probably Timeout threshold for calling Ollama API needs to be increased")
            pass
        except Exception as e:
            print("\n------- chunk id", j)
            print(f"Any other Error when applying LangExtract on document {filename_stem}, text block {j}: {e}")
            pass
    

    responses_all_docs.append(responses)




# %% [markdown]
# ## Saving

# %%
print("\n\n -------- Saving LangExtract output as jsonl and csv -------- \n")

responses_all_docs_unnested = list(chain(*responses_all_docs))


current_timestamp = datetime.today().strftime("%Y-%m-%d")


# check if file exists already
if os.path.exists(LX_OUTPUTS_FILEPATH):
    print(
        f"File {LX_OUTPUTS_FILEPATH.name} already exists. Renaming file to {LX_OUTPUTS_FILEPATH.stem}_{current_timestamp}.jsonl."
    )
    lx_response_filename_timestamped = (
        f"{LX_OUTPUTS_FILEPATH.stem}_{current_timestamp}.jsonl"
    )
    lx.io.save_annotated_documents(
        responses_all_docs_unnested,
        output_name=lx_response_filename_timestamped,
        output_dir=LX_OUTPUTS_FILEPATH.parent,
    )
else:
    print(f"Saving LangExtract output to {LX_OUTPUTS_FILEPATH}")
    lx.io.save_annotated_documents(
        responses_all_docs_unnested,
        output_name=lx_response_filename,
        output_dir=LX_OUTPUTS_FILEPATH.parent,
    )

# create dataframe of LX response
df_respones = pd.DataFrame(
    columns=[
        "infrastructure_type",
        "damage",
        "geolocation",
        "citation_id",
        "attributes",
        "name",
        "char_span",
        "token_span",
    ]
)
for i in range(len(responses_all_docs_unnested)):
    for j in responses_all_docs_unnested[i].extractions:
        try:
            ci_impact_dict = {
                "infrastructure_type": (
                    j.extraction_text
                    if j.extraction_class == "infrastructure_type"
                    else None
                ),
                "damage": j.attributes.get("damage", None),
                "geolocation": j.attributes.get("geolocation", None),
                #    "name": j.attributes.get("name", None),
                "attributes": j.attributes,
                "citation_id": responses_citations[i],
                "char_span": j.char_interval if hasattr(j, "char_interval") else None,
                "token_span": (
                    j.token_interval if hasattr(j, "token_interval") else None
                ),
            }
        except Exception as e:
            print(
                f"Error processing extraction {j} in document {responses_citations[i]}: {e}"
            )
            continue  # with next extraction

        df_respones = pd.concat(
            [df_respones, pd.DataFrame([ci_impact_dict])], ignore_index=True
        )


if os.path.exists(LX_OUTPUTS_DF_FILEPATH):
    print(
        f"File {LX_OUTPUTS_DF_FILEPATH.name} already exists. Renaming file to {LX_OUTPUTS_DF_FILEPATH.stem}_{current_timestamp}.csv."
    )
    LX_OUTPUTS_DF_FILEPATH = (
        LX_OUTPUTS_DF_FILEPATH.parent
        / f"{LX_OUTPUTS_DF_FILEPATH.stem}_{current_timestamp}.csv"
    )
    df_respones.to_csv(LX_OUTPUTS_DF_FILEPATH, index=False)

else:
    print(f"Saving LangExtract DF output to {LX_OUTPUTS_DF_FILEPATH}")
    df_respones.to_csv(LX_OUTPUTS_DF_FILEPATH, index=False)

print("\n\n -------- Finished LangExtract processing -------- \n")


# %%
df_respones



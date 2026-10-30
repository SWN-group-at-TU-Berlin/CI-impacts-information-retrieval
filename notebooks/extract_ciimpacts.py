# %%
# %%
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent     # .../ci-impacts-vllm/notebooks
PROJECT_ROOT = SCRIPT_DIR.parent                  # .../ci-impacts-vllm
os.chdir(PROJECT_ROOT)                 # "./ner_patterns.jsonl", "./spacy_model_pipeline", "./interim_results_debug" resolve in the project root
sys.path.insert(0, str(PROJECT_ROOT))  # makes "import src..." work from anywhere

use_geollama = False
RUN_EXAMPLE = False      # True: run the single text2 example graph before the document loop
# RUN_EVALUATION = False   # True: run the evaluation cells after the extraction
test_mode = True



import os
import sys
# import subprocess
import re
import time
from glob import glob
from pathlib import Path
import gc
from typing import List, Dict, Tuple, Optional, Union, TypeVar
# from io import StringIO
import json
import ast

# from tqdm import tqdm
from lxml import html   
import urllib3
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
import geonamescache
# import pyarrow as pa
# import pyarrow.parquet as pq
from huggingface_hub import login

from lingua import Language, LanguageDetectorBuilder
from transformers import AutoTokenizer
import spacy
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
# from docling_core.types import DoclingDocument
# from docling_core.types.doc import DocItemLabel
# from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, DocItem
# from hierarchical.postprocessor import ResultPostprocessor
# from langchain_docling.loader import ExportType
# from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import (
#     PdfPipelineOptions,
#     AcceleratorOptions,
#     AcceleratorDevice,
# )
# from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker


# %%
from src.settings import settings as s
from src.document_cleaning import extract_citation_info
import src.translation_model as tm
# import src.extraction_model as em
import src.postprocess as pp
import src.utils as u



# %%
import torch
print(torch.cuda.is_available())
# print(torch.cuda.device_count())  # should give 2
# print(torch.cuda.get_device_name())
# print(torch.cuda.get_device_properties(0))
# print(torch.cuda.get_device_properties(1))
# print(torch.cuda.get_device_capability())
# print(torch.cuda.get_arch_list())  # ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
# print(torch.__version__)
# print(torch.version.cuda)


# %%


# %%

# login to HF
from dotenv import load_dotenv
load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
if token:
    login(token=token)
else:
    print("No HUGGINGFACE_TOKEN set, skipping HF login (public models still load)")
        


# NOTE. disabled batch size as OOM for CUDA despite chunkwise memory cleaning, nvtop to find best batchsize
BATCH_SIZE = s.BATCH_SIZE  # max for nvidia GPU

# torch.manual_seed(42)

#  automatic linebreaks and multi-line cells.
pd.set_option("display.max_colwidth", 100000)
pd.set_option("display.colheader_justify", "left")


# clean up before applying CUDA
# gc.collect()
# torch.cuda.empty_cache() 
print(torch.cuda.memory_reserved() / 1e9)
torch.no_grad()


### LOGGER

# currently fetch only PydanticSerializationUnexpectedValue warnings 
# See, discussion in https://github.com/openai/openai-python/issues/2872
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")



## set path variables
DOCS_DIR = Path(s.PATH_DATA / "text_sources/")
# DOCS_DIR = Path("/home/anna/_CLUSTERFOLDER/data/text_sources/")
PARSED_TEXT_DIR = Path(s.PATH_DATA / "parsed_documents/")
NER_PATTERNS_FILEPATH = Path(s.NER_PATTERNS_FILEPATH)
LLM_OUTPUTS_DIR = Path(s.PATH_DATA / "llm_outputs/")
geollama_OUTPUTS_DIR = Path(s.PATH_DATA / "geollama_outputs/")

os.makedirs(PARSED_TEXT_DIR, exist_ok=True)
os.makedirs(s.PATH_LLM_DATA, exist_ok=True)
os.makedirs(geollama_OUTPUTS_DIR, exist_ok=True)


# CI GEO pairs
CI_GEO_FILEPATH = Path( s.PATH_DATA / s.CI_GEO_PAIRS_FILENAME)

## store LLM 1 response and prompt
OUTPUT_LLM1_FILEPATH =  Path(s.PATH_LLM_DATA / s.LLM_DATA_FILENAME)
OUTPUT_geollama_FILEPATH =  Path(s.PATH_LLM_DATA / "geollama_results.csv")




# %% [markdown]
# ### Set test mode

# %%

docs_list_sample = [
    Path(PARSED_TEXT_DIR, "Koks 2022 - Brief communication.pdf"),
    Path(PARSED_TEXT_DIR, "Khazai 2013 - Juni-Hochwasser 2013 in Mitteleuropa - Fokus Deutschland Bericht 2 Auswirkungen und Bewältigung.pdf"),
    Path(PARSED_TEXT_DIR, "Rozendaal 2021 - Infrabel_ Flood damage to railway track worth tens of millions of euros _ SpoorPro - incomplete.pdf"),
    Path(PARSED_TEXT_DIR, "Skoulding 2023 - Where are the fires in Italy today as temperatures rise to 47.6C on Sicily_ _ The Independent.pdf"),

    Path(PARSED_TEXT_DIR, "Lloyd's List 2024 - Port of Valencia reopens after devastating floods.pdf"),
    Path(PARSED_TEXT_DIR, "Containerlift 2024 - Valencia Port Resumes Operations Following Devastating Flooding in Spain - Containerlift.co.uk - Transport_Lifting_Shipping.pdf"), 
    Path(PARSED_TEXT_DIR, "ABC 2024 - Traffic jams and flight delays due to heavy rain and lightning storm in Malaga.pdf"),

    Path(PARSED_TEXT_DIR, "Karakatsani 2023 - Greece economy briefing The economic impact of the recent devastating floods in Greece.pdf"),
    Path(PARSED_TEXT_DIR, "European Investment Bank 2025 - Spain_ EIB lends €50 million to Iberdrola to rebuild and climate-proof flood-hit power infrastructure in Valencia.pdf"),
    Path(PARSED_TEXT_DIR, "Wilson 2024 - Flash floods in Spain sweep away cars, disrupt trains and leave several missing _ AP News.pdf"),     
    Path(PARSED_TEXT_DIR, "Wildhagen 2013 - Hochwasser_ Wie die Flut Unternehmen lahmlegt.pdf"),
    Path(PARSED_TEXT_DIR, "AFP 2022 - The_Vibes_Valencia Airport in Madrid briefly shut as lightning hits runway _ World _ The Vibes.pdf"),
    Path(PARSED_TEXT_DIR, "Artemis 2015 - PERILS finalises Storm Desmond UK flood loss estimate at £604m.pdf"),
    Path(PARSED_TEXT_DIR, "Brown 2010 - Economy feels chill as UK grinds to a halt _ The Independent.pdf"),
    #     # Path(PARSED_TEXT_DIR, "Diakakis 2020 - A systematic assessment of the effects of extreme flash floods on transportation infrastructure and circulation: The example of the 2017 Mandra flood.pdf"),
    Path(PARSED_TEXT_DIR, "EFE 2024 - The DANA storm, live_ The death toll rises to 158.pdf"),
    #     # Path(PARSED_TEXT_DIR, "Eurelectric 2006 - Impacts of Severe Storms on Electric Grids.pdf"),
    Path(PARSED_TEXT_DIR, "Euronews 2024 - Spain floods_ Death toll rises to 205 as nation braces for more rain .pdf"),
    Path(PARSED_TEXT_DIR, "Ferlita 2023 - Incendi in Sicilia, ecco cosa accade.pdf"),
    Path(PARSED_TEXT_DIR, "Fink 2004 - The 2003 European summer heatwaves and drought - synoptic diagnosis and impacts.pdf"),
    
    # Path(PARSED_TEXT_DIR, "Gilbody Dickerson 2024 - Spain floods_ At least 95 people killed including British man near Malaga _ World News _ Sky News.pdf"),
    # #     # Path(PARSED_TEXT_DIR, "Kadir 2014 - The Impact of Natural Disasters on Critical Infrastructures - A Domino Effect-based Study.pdf"),
    # Path(PARSED_TEXT_DIR, "Kaur 2025 - Authorities suspect arson in 17 wildfires across Dalmatian coast, Croatia - The Watchers.pdf"),
    # #     # Path(PARSED_TEXT_DIR, "Keller 2014 - Mapping Natural Hazard Impacts on Road Infrastructure—The Extreme Precipitation in Baden-Württemberg.pdf"),
    # #             # Path(PARSED_TEXT_DIR, "Koks 2019 - Understanding Business Disruption and Economic Losses Due to Electricity Failures and Flooding.pdf"),
    # #             # Path(PARSED_TEXT_DIR, "Korzilius 2021 Nach der Flut.pdf"),

    # Path(PARSED_TEXT_DIR, "Kettle 2020 - Storm Xaver over Europe in December 2013 Overview of energy impacts and North Sea events.pdf"),

    # Path(PARSED_TEXT_DIR, "Chamra 2006 - Flooding of the Prague metro during the August 2002 floods.pdf"),
    # # Path(PARSED_TEXT_DIR, "Hladny 2004 - August_2002_catastrophic_flood_in_the_Czech_Republic.pdf"),
    # Path(PARSED_TEXT_DIR, "Mitsakis 2014 - Impacts of high-intensity storms on urban transportation_ applying traffic flow control methodologies for quantifying the effects.pdf"),
    # Path(PARSED_TEXT_DIR, "Pescaroli 2017 - How Critical Infrastructure Orients International Relief in Cascading.pdf"),
    # Path(PARSED_TEXT_DIR, "Nieuwsblad 2021 - Geen drinkbaar water, geen gas, en geen elektriciteit_ 9.000 mensen moeten op zoek naar ander onderkomen.pdf"),

    # Path(PARSED_TEXT_DIR, "Treanor 2015 - Storm Desmond damage across Cumbria estimated at £500m _ Storm Desmond _The Guardian.pdf"),
    # # Path(PARSED_TEXT_DIR, "Koks 2022 - Brief communication.pdf"),

    # Path(PARSED_TEXT_DIR, "AEMET 2024 - ESTUDIO SOBRE LA SITUACIÓN DE LLUVIAS INTENSAS.pdf"),
    # Path(PARSED_TEXT_DIR, "Hladny 2004 - August_2002_catastrophic_flood_in_the_Czech_Republic.pdf"),
    # Path(PARSED_TEXT_DIR, "Nour 2011 - Damages caused by floods and flash-floods upon critical infrastructure.pdf"),
    # Path(PARSED_TEXT_DIR, "Fink 2009 - The European storm Kyrill in January 2007_ synoptic evolution, meteorological impacts and some considerations with respect to climate change.pdf"), 

#     #     # not part of valid set:
#     #     # Path(PARSED_TEXT_DIR, "Krausmann 2014 - STREST report on lessons learned from recent catastrophic events.pdf"), # > 1800 entries LLMv3.0 incl. hallucinations
]


## Test mode
if test_mode:
    search_path = docs_list_sample
    print("Test mode is ON. Using only a small sample of documents for testing.")
else:
    search_path = glob(str(Path(PARSED_TEXT_DIR, "*cleaned.jsonl")))

print(f"Using {len(search_path)} documents for processing.")



# %% [markdown]
# ###  Load spaCy language model

# ## RELOAD spacy pipeline
# nlp = spacy.load("./spacy_model_pipeline")
# NER_PATTERNS_FILEPATH = Path("./ner_patterns.jsonl/")

# # add CI_TYPE patterns to spacy nlp model pipeline
# config = {"spans_ke": None, "annotate_ents": True, "overwrite": False}
# ## see for more info: https://spacy.io/usage/rule-based-matching#entityruler
# ## NOTE EntityRuler is hidden inside .add_pipe()
# try:
#     ruler = nlp.add_pipe("span_ruler", config=config)
#     ruler.from_disk(NER_PATTERNS_FILEPATH)
# except ValueError:
#     print("SpanRuler already exists in pipeline.")
#     ruler = nlp.get_pipe("span_ruler")
#     ruler.from_disk(NER_PATTERNS_FILEPATH)


# # load NER patterns for CI types and their subgroups (needed for cleaning LLm response - STEP 1 )
ci_patterns = pd.read_json("./ner_patterns.jsonl/patterns", lines=True)



# %%


# %% [markdown]
# 

# %% [markdown]
# ## NER model

# %%
class NERModel():

    def __init__(self):
        self.nlp_model = self.initialize()
    

    def initialize(self):
        """ Return initialized NER model"""
  
        dir_model_pipeline="./spacy_model_pipeline"
        dir_ci_patterns=Path("./ner_patterns.jsonl")

        # LOAD spacy pipeline
        nlp = spacy.load(dir_model_pipeline)

        # add CI_TYPE patterns to spacy nlp model pipeline
        config = {"spans_key": None, "annotate_ents": True, "overwrite": False}
        ## see for more info: https://spacy.io/usage/rule-based-matching#entityruler
        ## NOTE EntityRuler is hidden inside .add_pipe()
        try:
            ruler = nlp.add_pipe("span_ruler", config=config)
            ruler.from_disk(dir_ci_patterns)
        except ValueError:
            print("SpanRuler already exists in pipeline.")
            ruler = nlp.get_pipe("span_ruler")
            ruler.from_disk(dir_ci_patterns)

        # load NER patterns for CI types and their subgroups (needed for cleaning LLm response - STEP 1 before continuing with STEP 2)
        ci_patterns = pd.read_json("./ner_patterns.jsonl/patterns", lines=True)

        return nlp


# %%


# %% [markdown]
# ## LangGraph: 

# %%
import ast

from typing import TypedDict, List, Dict, Optional, Literal
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, MessagesState, add_messages, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
import openai
from pydantic import BaseModel
from langchain.agents.middleware import wrap_tool_call, wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import ToolMessage
from langchain_core.tools import tool
from langchain.tools.tool_node import ToolCallRequest
from collections.abc import Callable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore



# %% [markdown]
# #### LangGraph prompts

# %%

###########################################################################
# Prompt 1
# Extract infrastructure assets
###########################################################################
is_case_prompt = ChatPromptTemplate.from_messages([
    ("system",
"""This is your first step for which you must carefully read the TEXT. 
Then, decide if the TEXT describes an impact or damage to critical infrastructure caused by a natural hazard.

ITEM TO EXTRACT:
 - "is_infrastructure_case": "Return a Boolean. If TEXT describes one or multiple impacts or damages to critical infrastructure caused by a natural hazard, then return <True>. If not, return <False>.


Return False when:
- Affected or damage infrastructure is mentioned only in general, i.e. without a link to one or multiple past hazard events
- Potential future impacts or damages to infrastructure are mentioned or when the Text is about preparations against a hazard


"""

    ), 
    ("human", "TEXT: {text}")
])
    

# An impact or damage to critical infrastructure occurs when critical infrastructure asset or infrastructure service is damaged through disruption, structural damage, failure or limited operability.



# **Important**
# * When an entry contains two or more infrastructure assets then keep them together. For instance: entry: "There is heavy traffic on the access roads from Juan Avenue to Plaza Manuel Azaña and to Blas Infante Avenue" --> "on the access roads from Juan Avenue to Plaza Manuel Azaña and to Blas Infante Avenue"
# * When an entry contains two or more infrastructure assets then split them in two. For instance: entry: "Road and railway infrastructure" -->  "Road infrastructure", "railway infrastructure"

asset_prompt = ChatPromptTemplate.from_messages([
    ("system",
"""Extract every damaged infrastructure asset or service mentioned in the TEXT. The damage has to be caused by a natural hazard.
Use the entries returned from the "get_infrastructure_entities" tool: decide based on the TEXT for each entry if it really refers to an infrastructure damaged or affected by a natural hazard. If this is the case, you should add the entry to the JSON object.  


ITEM TO EXTRACT:
- "List of infrastructure assets or services damaged by a natural hazard." 

Example:
ci_assets : ["roads", "railway infrastructure"]
"""),
    ("human", "TEXT: {text}" )
])


asset_information_prompt = ChatPromptTemplate.from_messages([
    ("system",
"""
You are an information extraction agent for critical infrastructure.
Your task is to combine:

1. the original TEXT
2. the infrastructure entities detected by the NER tool

and produce a structured list of infrastructure cases.

For EACH infrastructure entity detected by the tool, identify:

- the infrastructure asset or service ("ci_asset")
- the number of affected infrastructure asset or service ("ci_asset_amount")
- the name or designation of the infrastructure asset ("ci_asset_name")
- its corresponding location ("location")
- its corresponding damage category ("damage")


IMPORTANT:
- Use the infrastructure entities returned by the NER tool as the candidate infrastructure entities.
- Do not invent additional infrastructure entities.
- Use the original TEXT to determine the location and damage associated with each infrastructure entity ("ci_asset"). Also, for each entity determine if a number is mentioned, i.e., determine how many infrastructure assets or services are affect for each infrastructure entity. 
- Assign the correct location and damage to the corresponding asset.
- Assign the correct number of assets or to the corresponding asset (e.g. "more than 68" when the text mentions that more than 68 hospitals were damaged).
- Prioritize the most specific location of each infrastructure (e.g., street names, neighborhoods, towns over cities, cities over regions, regions over countries).
- Return the location of each infrastructure as detailed as possible, eg. that it can be identified on a map
- You should not consider adjectives (e.g. "German", "English", "Wallonian") as information about the location. Instead derive the correct location based on the TEXT or the adjectives (e.g., "Germany", "United Kingdom", "Wallonia").
- **Only for transportation infrastructure:** When the specific name of an ASSET is mentioned (e.g. Juan Avenue, intersection to Maria Statue, highway A-1, B510, Berlusconi Airport) and no location information can be found, then consider the specific name of this asset as the location. But do not use road types as location names, such as "German highway", "national road", etc.
- **Only for transportation infrastructure:** when an ASSET refers to two or more locations then extract the locations together. For instance: text:"There is much damage on the access roads from Juan Avenue to Plaza Manuel Azaña" --> location:["from Juan Avenue to Plaza Manuel Azaña"]
- If no location is explicitly stated for an asset, return None for "location".
- Assign ONE general damage category ("damage") to each infrastructure entity, based on the damage, failure, or disruption the TEXT reports for it. If several categories apply, use the most severe one (the first one in the following list). Damage categories:
""" + pp.describe_damage_categories() + """
- If no damage or disruption is explicitly stated for an asset, return None for "damage".
- If no information about the number of infrastructure assets or services is explicitly stated for an infrastructure entity ("ci_asset"), return None for "ci_asset_amount".
- When a proper name or designation of an infrastructure entity is explicitly mentioned in the TEXT (e.g. road names such as "A1", "B510", "Gates Avenue", or facility names such as "Braunschweig chemical plant", "Berlusconi Airport"), return it as "ci_asset_name" and preserve its original wording. Do not use generic infrastructure types (e.g. "highway", "national road", "chemical plant") as "ci_asset_name".
- If no name or designation is explicitly stated for an infrastructure entity ("ci_asset"), return None for "ci_asset_name".
- Do not infer information that is not explicitly mentioned in the TEXT.



Example Input:
text: Similar as in Germany, bridges were often flooded, said the General Officer of the EU. But also sever impacts occurred in other places, for example, road and railway infrastructure were damaged or destroyed in Western Germany. In total, around 50 bridges were destroyed in Germany. The highway A1 near Cologne was covered by debris.

Example Output:
[InfrastructureCase(ci_asset="bridges", ci_asset_amount=None, ci_asset_name=None, location=None, damage="structural_damage"), InfrastructureCase(ci_asset="road and railway infrastructure", ci_asset_amount=None, ci_asset_name=None, location="Western Germany", damage="structural_damage"), InfrastructureCase(ci_asset="bridges", ci_asset_amount="around 50", ci_asset_name=None, location="Germany", damage="structural_damage"), InfrastructureCase(ci_asset="A1 highway", ci_asset_amount=None, ci_asset_name="A1", location="Cologne", damage="obstruction"),]

"""),
    ("human", """
TEXT: {text}
INFRASTRUCTURE ENTITIES DETECTED BY NER: {entities}
""")
])


###########################################################################
# Conditional Prompt
# Asset finalize
###########################################################################

# asset_finalize_prompt = ChatPromptTemplate.from_messages([
#     ("system",
# """Take the MESSAGE and return the list of items into a JSON object. 
# **Important**: DON'T add or remove any items.

# Return a JSON object.

# Example:
# ci_asset: ["roads", "railway infrastructure"]
# """),
#     ("human", "MESSAGE: {messages}")
# ])

 

###########################################################################
# Prompt 2
# Extract locations
###########################################################################

# For each infrastructure in the list ASSETS, you should find its location mentioned in the TEXT. 


# The list of locations must be as long as LENGTH_ASSETS.


location_prompt = ChatPromptTemplate.from_messages([
    ("system", """For every infrastructure asset or service returned by ASSETS determine its location by reading the TEXT.
Store each identified location as an entry in the "location" list.
DON'T create new items about locations that were not there.

ITEM TO EXTRACT:
- "location": "List the location of each infrastructure asset or service mentioned in ASSETS."         



**Important:** 
* Prioritize the most specific location of each infrastructure (e.g., street names, neighborhoods, towns over cities, cities over regions, regions over countries).
* **Only for transportation infrastructure:** When the specific name of an ASSET is mentioned (e.g. Juan Avenue, intersection to Maria Statue, highway A-1, B510, Berlusconi Airport) and no location information can be found, then consider the specific name of this asset as the location.
* **Only for transportation infrastructure:** when an ASSET refers to two or more locations then extract the locations together. For instance: text:"There is much damage on the access roads from Juan Avenue to Plaza Manuel Azaña" --> location:["from Juan Avenue to Plaza Manuel Azaña"]
* If you cannot find information about the location of an infrastructure, then return a "NAN" value for this entry in the "location" list. 
* The location of the affected infrastructure must be explicitly mentioned in the TEXT.


**Exclude:** 
* You should not consider adjectives (e.g. "German", "English", "Wallonian") as information about the location. Instead derive the correct location based on the TEXT or the adjectives (e.g., "Germany", "United Kingdom", "Belgium").
* **Future/Warning:** Locations listed as "at risk", "under warning", "may face", "could be affected," or "preparing for" a hazard event. 
* **Historical/Contextual:** Locations mentioned only as background information or prior events (e.g., "This follows flooding in [Location X] last month," or "[Location Y] had **already** experienced flooding by this period").


Example Input:
text: Some road and railway infrastructure were damaged in Western Germany. The event caused damages in entire Europe, said the General Officer of the EU. Bridges were often damaged.
asset: ["road and railway infrastructure", "bridges"]

Example Output:
location: ["Western Germany", "NAN"]

"""),
("human",
"""
TEXT: {text}

ASSETS: {assets}

""")
])


# # Return a JSON object.

# "location": [
# {{
#     "roads":"Western Germany", 
#     "railway infrastructure":"Western Germany", 
#     "waterways":"NAN", 
#     <assets_nth>: <location_nth>,
# }}
# ]
# **Important:** 
# Only if you identified an affected critical infrastructure asset or object.
# - Identify the specific location of the affected infrastructure that the TEXT explicitly states that this infrastructure asset or service was affected. Return "NAN" if the TEXT does not mention a location of the affected infrastructure.
# - If no information about the location of an affected infrastructure is found, then return a "NAN" value. 
# - The location of the affected infrastructure must be explicitly mentioned in the TEXT.
# - In case of transport infrastructure (e.g. railway lines, roads or highways) the location can consists of two or more towns, cities or regions (e.g. "railway line between City A and City B", "highway from City X to City Y", "highway between Valencia and Malaga"). In this case, create a new entry for each town, city or region.
# - Prioritize the most specific location that the TEXT explicitly mentions for each of the affected infrastructure assets (e.g., street names, neighborhoods, towns over cities, cities over regions, regions over countries).  
# - If a specific location lacks detailed background information (e.g., "Main Street" without a city), add the broader context (e.g., "Main Street, City A").  
# - Some locations may span multiple words.


# **Exclude:** 
# * You should not consider adjectives (e.g. "German", "English", "Wallonian") as information about the location. Instead derive the correct location based on the TEXT (e.g., "Germany", "United Kingdom", "Belgium").
# * **Future/Warning:** Locations listed as "at risk", "under warning", "may face", "could be affected," or "preparing for" a hazard event. 
# * **Historical/Contextual:** Locations mentioned only as background information or prior events (e.g., "This follows flooding in [Location X] last month," or "[Location Y] had **already** experienced flooding by this period").
# * **Only extract locations that are part of the specific incident described in the document.** 



# Example:
# [
# {{
# "location": ["Western Germany", "Western Germany", "NAN"]
# }}
# ]
# ci_asset: ["roads","railway infrastructure", "bridge"]

ci_loc_verify_prompt = ChatPromptTemplate.from_messages([
    ("system", """For each pair "ci_asset" - "location" in EXTRACTED_CASES you should evaluate and improve your answer based on the information mentioned in ENTITY_RECOGNITION.
ENTITY_RECOGNITION returns two potential locations ("geo_entity", "geo_2_entity") of each affected critical infrastructure ("ci_entity").
Keep in mind, that the location information returned from ENTITY_RECOGNITION are just for guidance. You still need to make sure to only identify the specific location of each affected infrastructure that the TEXT explicitly mentions that this infrastructure asset was affected at this specific location.

If you do not find specific information about the location of an affected infrastructure, then return a "NAN" value for this field. 


**Important:** 
DO NOT consider the information from ENTITY_RECOGNITION when the "ci_entity" is mentioned in regard to warnings or in a historical context. In this case return "NAN" values for the fields "ci_asset" and "location":
Exclude:
* **Future/Warning:** Locations listed as "at risk," "under warning," "may face," "could be affected," or "preparing for" a hazard event. 
* **Historical/Contextual:** Locations mentioned only as background information or as prior events (e.g., "This follows flooding in [Location X] last month," or "[Location Y] had **already** experienced flooding by this period"). **Only extract locations that are part of the specific incident described in the document.** 
"""),
    ("human",
"""
TEXT: {text}

EXTRACTED_CASES: {extracted_cases}

ENTITY_RECOGNITION: {entity_recognition}
""")
])
###########################################################################
# Prompt 3
# Extract damage type
###########################################################################

# LENGTH_ASSETS: {assets_length}
# The list of damages must be as long as LENGTH_ASSETS.


damage_prompt = ChatPromptTemplate.from_messages([
    ("system",
"""For every infrastructure asset or service returned by ASSETS determine the damage type.
Use for guidance the keywords from the following list to describe the damage type, but also create a new one when you cannot find it in the list: 
[affected, damaged, severely damaged, highly damaged, disrupted, blocked, destroyed, largely destroyed, heavily destroyed, partly closed, closed, outage(s), blackout, contaminated, polluted, flooded, inundated, fallen tree(s), broken tree(s), overturned tree(s), uprooted tree(s), felled tree(s), debris, debris removal, debris on, water on, dam failure(s), dyke break(s), dike break(s), breached dike(s), broken embankment(s), closure(s), derailed, impassable, traffic jam(s), traffic disruption(s), traffic congestion(s), indirectly affected, flight delay(s), train delay(s), delay(s), train failure(s), failure(s), cancellation(s), canceled, re-established, reestablished, restored, reconnected, rebuild, repaired]

If no information of a damage type is found, then return a "NAN" value.


ITEM TO EXTRACT:
- "damage": "List the damage type of each affected infrastructure."


Return a JSON object.


Example:

ci_asset:["roads","railway infrastructure"]
damage: ["blocked", "flooded"]

"""),
    ("human",
"""
TEXT: {text}

ASSETS: {assets}

""")
])

# Example:
# [
# {{
# ci_asset:["roads","railway infrastructure"],
# location: ["Western Germany", "Western Germany"],
# damage: ["blocked", "flooded"],
# }}
# ]
# location: ["Western Germany", "Western Germany"]

###########################################################################
# Prompt 4
# Merge
###########################################################################
# Nothing else.




###########################################################################
# Prompt 5
# Check cascading effects
###########################################################################

verification_prompt = ChatPromptTemplate.from_messages([
    ("system",
"""
You receive extracted infrastructure damage cases.

For EACH case determine whether the text states that the damage affected another infrastructure asset or infrastructure service (field: "secondary_effects"), return <true>.
If no return <false>.

If yes, identify the affected infrastructure (field: "target_asset"), else return "NAN".


Example Output:
secondary_effects:[ false, false, true, false]
target_asset:["NAN","NAN", "gas supply", "NAN"]

Only use information explicitly stated.
Do not infer.


"""),
    ("human",
"""
TEXT: {text}

CASES: {cases}
""")
])


# %%
## GEOLLM prompt


###########################################################################
# Node GEOLLM 
###########################################################################


enrich_locations_prompt = ChatPromptTemplate.from_messages([
    ("system",
"""
You are a geographic metadata enrichment agent.

The first extraction models have already identified the following infrastructure information:
- TEXT: {text}
- LOCATION: {locations}


Your ONLY task is to obtain geographic coordinates and OSM metadata for the extracted LOCATION using "get_coordinates_and_meta_info" tool.
Pass the TEXT information as input to "get_coordinates_and_meta_info" tool.
Then use the entries returned from the tool and check which one corresponds to the passed LOCATION. The best matched entry is where the entry["loc_name"] name corresponds with the LOCATION name. Return this entry (loc_name, geometry, osm_key, osm_value ) as your response.

"""),
    ("human", """ 
LOCATION: {locations}
   
TEXT: {text}
    """)
])
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


# %% [markdown]
# #### Tools - NER + GEoLLM

# %%

###########################################
# Tools
###########################################

nlp_model = NERModel().nlp_model

@tool
def get_infrastructure_entities(text:str) -> List[str]| None:
    """This tool detects infrastructure assets and services from text by using a spaCy model to conduct Named Entity Recognition (NER)""" 

    # remove common misleading CI keywords 
    # ("dam" - "damage.*", "train"- "training") as NER patterns not recognize stop-char "\b" for any reason
    # but keep when rest of entity is CI, eg, "damaged rails", 
    regex = re.compile(r"\b[Dd]amage\w*|\b[Tt]raining\w*|\b[Cc]ommunication\w*")
    text_clean = re.sub(regex, "", text).replace("  ", " ")
    

    # get ci entities
    nlp_chunk = nlp_model(text_clean)
    ci_entities = [ent for ent in nlp_chunk.ents if ent.label_ in ["CI_TYPE", "FAC"]]
    
    ## cleaning:
    # remove common errorneous CI keywords ("dam" - "damage.*", "train"- "training") as NER patterns not recognize stop-char "\b" for any reason
    regex = re.compile(r".*damage.*|.*training.*|\b[Cc]ommunication")
    ci_entities = [str(i) for i in ci_entities if not regex.match(str(i))]
    
    if ci_entities:
        return ci_entities
        
    return None
    # return f"Following entities about critical infrastructure were found in the text: {ci_entities}."


# @tool
def get_infrastructure_location_pairs(text:str) -> List[str] | None:
    """This tool detects infrastructure assets and services from text by using a spaCy model to conduct Named Entity Recognition (NER)""" 
    
    
    # remove common misleading CI keywords 
    # ("dam" - "damage.*", "train"- "training") as NER patterns not recognize stop-char "\b" for any reason
    # but keep when rest of entity is CI, eg, "damaged rails", 
    regex = re.compile(r"\b[Dd]amage\w*|\b[Tt]raining\w*|\b[Cc]ommunication\w*")
    text_clean = re.sub(regex, "", text).replace("  ", " ")
    
    nlp_model = NERModel().nlp_model
    nlp_chunk = nlp_model(text_clean)
    
 
    # GET CI and related LOCs entities
    all_ents = [ent for ent in nlp_chunk.ents]

    ## cleaning:
    # remove common errorneous CI keywords ("dam" - "damage.*", "train"- "training") as NER patterns not recognize stop-char "\b" for any reason
    regex = re.compile(r".*damage.*|.*training.*")
    all_ents = [i for i in all_ents if not (regex.match(str(i)) and i.label_ in ["CI_TYPE", "FAC"])]
    
    # remove all FAC which are not roads, airport or port names
    print("FAC before removal", [i for i in all_ents if i.label_ in ["FAC"]])
    regex_roadnames = re.compile(r".*[A-Z]{1,3}[0-9]{1,3}.*|.*[A-Z]{1,3}-[0-9]{1,3}.*")  # matches: A-75, BE105
    regex_others = re.compile(r".*airport.*|.*port.*|")
    all_ents = [i for i in all_ents if ((regex_roadnames.match(str(i)) or regex_others.match(str(i))) and i.label_ in ["FAC"])]
    
    print("FAC after removal", [i for i in all_ents if i.label_ in ["FAC"]])

    # init dfs for interim results for each chunk
    # TODO make as pydantic class with fixed attributes
    df_ci_geo_chunk = pd.DataFrame(
        columns=[
            "chunk_text",
            "ci_entity",
            "ci_entity_label",
            "geo_entity",
            "geo_2_entity",
            "geo_entity_label",
            "token_distance",
            "token_2_distance",
        ]
    )    

    # iterate over all entities within text
    for ent_idx in range(len(all_ents)):
        # when entity is CI_TYPE
        if all_ents[ent_idx].label_ in ["CI_TYPE", "FAC"]: 
            ci_idx = ent_idx

            ## .. calculate distances between CI_TYPE entity and all GEO entities based on index position
            distance_list = []
            idx_in_chunk = []
            try:
                for ent_idx in range(len(all_ents)):

                    # TODO calc distances between CI_TYPE ~ GEO entities based on word numbers and not entities (ie tokens)
                    if all_ents[ent_idx].label_ in ["GPE", "LOC"]:

                        ## check that GPE,LOC are not countries (too coarse info CI-GEO pair)
                        # of GPE/LOC is country -> proceed with next GPE/LOC 
                        # if all_ents[ent_idx].text in ci_geo_countries:
                        #     continue

                        geo_idx = ent_idx
                        dist_ent_pair = np.abs(ci_idx - geo_idx)
                        distance_list.append(dist_ent_pair)
                        idx_in_chunk.append((ent_idx))

                # for each calc of the distance of a ci-loc pair, sort distance list always in ascending order 
                closest_pair_idx = np.argsort(distance_list)[0]#np.argmin(distance_list)  # idx of closest GEO entity
                
                # handle case when only 1 location exists in text
                closest_2nd_pair_idx = None
                if len(distance_list) > 1: 
                    closest_2nd_pair_idx = np.argsort(distance_list)[1]                    
                
                distance_closest_pair = distance_list[closest_pair_idx]
                # print("If token dist > 5 then probably no link. See how often ths is th case")

                # # consideer only close CI-LOC pairs (based on number of all_ents)
                # # NOTE: distance is based on number of entities (all_ents) (( - which are in turn tokens but not comprise all existing in a chunk!))
                threshold = 4
                if distance_closest_pair > threshold:
                    # print(
                    #     f""" Token distance is too large between CI_TYPE/FAR and next GEO entity which is of {distance_closest_pair} [token distance] > {threshold} [max. token distance] """
                    # )
                    continue

                ## write as dict entry incl chunk_id, ci_entity, geo_entity, distance
                result_dict = {
                    "chunk_text": text,
                    "ci_entity": all_ents[ci_idx].text,
                    "ci_entity_label": all_ents[ci_idx].label_,
                    "geo_entity": all_ents[idx_in_chunk[closest_pair_idx]].text,
                    "geo_2_entity": all_ents[idx_in_chunk[closest_2nd_pair_idx]].text if (len(distance_list)>1 and distance_list[closest_2nd_pair_idx]<threshold) else None,
                    "geo_entity_label": all_ents[idx_in_chunk[closest_pair_idx]].label_,
                    "token_distance": distance_closest_pair,
                    "token_2_distance": distance_list[closest_2nd_pair_idx], # np.sort(distance_list)],
                }                        
                df_ci_geo_chunk = pd.concat(
                    [  df_ci_geo_chunk, pd.DataFrame([result_dict])], ignore_index=True
                )
            except (IndexError, NameError) as e:
                # print(e)
                continue
    
            # ## post-process of DF CI-GEO pairs for each chunk
            # df_ci_geo_chunk = df_ci_geo_chunk.drop_duplicates(
            #     subset=["ci_entity", "geo_entity","chunk_text"])
            # ## return only closest pairs
    
    return df_ci_geo_chunk[["ci_entity","geo_entity","geo_2_entity"]]# ,"token_distance", "token_2_distance"]]




# %%
# # text = "A few national highways were partly flooded (e.g. the A76 in both directions) or briefly closed (&lt; 3 d) because of the potential of flooding. Most likely due to relative low-flow velocities, damage to Dutch national road infrastructure was limited. Several railway sections were closed (e.g. the railway section between Maastricht and Liége) and some damage occurred to the railway infrastructure, in particular to the electronic 'track circuit' devices and saturated railway embankments (Prorail, 2021).   "
# text = "Similar to Germany, severe damage had been observed to the gas network. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021). In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021)."
# # print(text)
# get_infrastructure_location_pairs(text)#[["ci_entity", "geo_entity","geo_2_entity","token_distance", "token_2_distance"]]

# %% [markdown]
# #### LangGraph OutputSchema and Graph

# %%
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
    ci_asset: str =  Field(
        default=None,
        description="Infrastructure assets or services damaged by a natural hazard.")
    ci_asset_amount: str | None = Field(
        default=None,
        description=("Number of affected infrastructure assets or services (ci_asset) explicitly reported in the text, if any. Preserve the original wording. Example: In the villages around Berlin around 100 roads were damaged. ci_asset:'roads', ci_asset_amount:'around 100'"
    ))
    ci_asset_name: str | None = Field(
        default=None,
        description=("Proper name or designation of the affected infrastructure asset (ci_asset) explicitly mentioned in the text, if any. Preserve the original wording. "
                     "Examples: 'A1', 'B510', 'Gates Avenue', 'Braunschweig chemical plant', 'Berlusconi Airport'. "
                     "Example: The A1 highway was flooded near Cologne. ci_asset:'A1 highway', ci_asset_name:'A1'"
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



###########################################
# Tools
###########################################

nlp_model = NERModel().nlp_model

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


# %%

###########################################################################
# LLM
###########################################################################


# init model for CI base-impacts, served by vLLM on the HPC (see scripts/serve_vllm_gptoss.sbatch)
llm = ChatOpenAI(
    model=s.LLM1_MODEL,
    base_url=s.LLM1_BASE_URL,
    api_key=s.VLLM_API_KEY,
    temperature=0,
    seed=42,
    timeout=300,
    max_tokens=8192,  # counts reasoning + answer: prompt + max_tokens must be <= vLLM --max-model-len (16384)
    reasoning_effort="low",  # gpt-oss: low | medium | high
)


class LengthLimitError(RuntimeError):
    """gpt-oss used up max_tokens (mostly for reasoning) before returning the structured answer"""


def structured_llm(schema):
    """llm.with_structured_output via tool calling, raising LengthLimitError when max_tokens is reached.
    With method="function_calling" langchain returns None instead of raising openai.LengthFinishReasonError,
    which would make the `while result is None` retries loop on the same (deterministic) overrun."""

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

    return llm.with_structured_output(schema, method="function_calling", include_raw=True) | RunnableLambda(_parse)

# init LLM-as-a-judge (step 3)
llm_judge = ChatOpenAI(
    model=s.JUDGE_MODEL,
    base_url=s.JUDGE_BASE_URL,
    api_key=s.VLLM_API_KEY,
    temperature=0,
    seed=42,
    timeout=300,
    max_tokens=8000 # adapt in respect to GEMMA_MAX_LEN=12288, TODO try out to increase with lower batch size, for h200: 16384,  # the judge (reasoning model) needs tokens for reasoning + JSON answer; vLLM --max-model-len must be >= prompt + max_tokens
)
# NOTE judge replaced: nemotron-3-super-120b-a12b needs 2-4x 80 GB GPUs; gemma-4-31B-it (reasoning very good) fits on 1 GPU, on NIM it timed out (2026-10-18)
# NOTE "openai/gpt-oss-20b"  # gpt-oss seems to have better reasoning for edge cases partly (germay+beligum, bridges) than "meta/llama-3.3-70b-instruct"



####################################
# Create agents: 
#  - Bind Tools and  OutputSchema to LLM
####################################


# bind tool to graph-node(s) of 1st model
tools = [get_infrastructure_entities]
tool_node = ToolNode(tools)
llm_tool =  llm.bind_tools(
    tools, 
    tool_choice="any", 
    parallel_tool_calls=False
)

# # bind tool two 2nd model
# geo_tools = [get_coordinates_and_meta_info]
# geo_tool_node = ToolNode(geo_tools)
# llm_enrichment_tools = llm_enrichment.bind_tools(
#     geo_tools,
#     tool_choice="any",
#     parallel_tool_calls=False,
# )



# %%
## create agent


# ## create agent
# llm_tool = create_agent(
#     model=llm_tool, 
#     # system_prompt="You are a helpful assistant",
#     middleware=[handle_tool_errors],
# )


# %% [markdown]
# ##### Graph components

# %%
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

    result = llm_tool.invoke(messages)

    # if not result.tool_calls:
    #     raise RuntimeError(
    #         "LLM did not call get_infrastructure_entities."
    #     )
    i = 0    
    while not result.tool_calls or len(result.tool_calls) == 0:
        print(f"⚠️ No tool calls were returned by the model.Trying again [{i}]")
        result = llm_tool.invoke(messages)        
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

    # state["secondary_effects"] = result.secondary_effect
    # state["affected_asset_or_services"] = result.secondary_effects
    # state["final_output"] = result
    # return state


# %%


# %% [markdown]
# ##### Cascading components (subagents)

# %%
prompt_has_cascade = ChatPromptTemplate.from_messages([
    ("system","""
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


"""
    ), 
    ("human", """

TEXT: {text}

CI_DAMAGES: {ci_damages}

""")
])


# Determine for every damaged infrastructure asset or service (ASSETS) whether it causes a downstream impact according to the TEXT.

# For every infrastructure asset or service returned by ASSETS determine whether the following TEXT mentions also one or multiple cascading impacts caused by the failure or disruption of this infrastructure, i.e. determine for each infrastructure failure if it led to one or multiple downstream impacts.
# A cascading impact occurs when: 
# An a single infrastructure asset or service (ASSETS) is damaged or disrupted, AND causes a further impact, such as economic costs, affects people or households, or affects another infrastructure asset or service.


#     Determine for each infrastructure failure (DAMAGED_ASSET) if the TEXT mentioned that it led to one or multiple downstream impacts. 
# A downstream impact occurs when: A single infrastructure asset or service is damaged or disrupted (DAMAGED_ASSET), AND caused a further impact, e.g., caused economic costs, affects people or households, or affects another infrastructure asset or service.

# ITEM TO EXTRACT:
# - "has_cascading_impacts": "List of booleans indicating for each affected infrastructure (DAMAGED_ASSET) if it caused downstream impacts."

# Example:
# ci_asset:["roads","railway infrastructure"]
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
    
    #     
    # has_cascade_list = []
    # for case in ci_cases:
        # result = llm_structured.invoke([
        #     HumanMessage(content=prompt_has_cascade.format(
        #         text=state["text"],
        #         damaged_infrastructure=case,
        # ))])
    #     has_cascade_list.append(result)
    # print("has_cascading_impacts", has_cascade_list)

    # return {"has_cascading_impacts": has_cascade_list}






# %%
economic_impact_prompt = ChatPromptTemplate.from_messages([
    ("system","""
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
CI_DAMAGES: ci_asset="roads", location="Brandenburg", damage="flooded"
source_asset: roads
source_location: Brandenburg
impact_origin: repair_recovery_cost
impact_type: repair cost
amount: 1,000 million EUR
unit: EUR

TEXT: Similar to Germany, severe damage had been observed to the gas network. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with cost reaching more than 5 million Euros. In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021).
CI_DAMAGES: ci_asset="gas supply", location="Liége, Chaudfontaine and Pepinster (Belgium)", damage="recovered"
source_asset: gas supply
source_location: Liége, Chaudfontaine and Pepinster (Belgium)
impact_origin: repair_recovery_cost
impact_type: recovery cost
amount: more than 5 million Euros
unit: EUR

TEXT: The flight delays and further transportation disruption leads to estimated costs of above 5.000 £ in UK.
CI_DAMAGES: ci_asset="flights", location="UK", damage="delays"
source_asset: roads
source_location: UK
impact_origin: downstream_economic_consequence
impact_type: economic loss
amount: above 5.000 £
unit: GBP

TEXT: The VW factory was closed for 3 days due to flooded roads leading to costs of 1,000 EUR. However, the entire costs of the flood event in the region of Brandenburg caused costs of 5,0000 million Euros.
CI_DAMAGES: ci_asset="roads", location=None, damage="flooded"
source_asset: roads
source_location: None
impact_origin: downstream_economic_consequence
impact_type: economic loss
amount: 1,000 EUR
unit: EUR

TEXT: Damaged transport network lead to production impacts for companies in Germany.
CI_DAMAGES: ci_asset="transport network", location="Germany", damage="damaged"
source_asset: transport network
source_location: Germany
impact_origin: downstream_economic_consequence
impact_type: production impacts
amount: None
unit: None


TEXT: Because the engine plant in Chemnitz could not bring its delivery over the streets because they were blocked or blocked by flood rescue forces. Also rear axle modules, which are pre-assembled by the VW-Hoflogistiker Schnellecke in neighbouring Glauchau, could not be loaded in the direction of Zwickau for this reason.
CI_DAMAGES: ci_asset="streets", location="Chemnitz", damage="blocked"
source_asset: streets
source_location: Chemnitz
impact_origin: downstream_economic_consequence
impact_type: transportation disruption
amount: None
unit: None

TEXT: Asset damages for the German railway provider are estimated to be around 20 000 EUR.
CI_DAMAGES: ci_asset="railway assets", location=None, damage="delays"
source_asset: railway assets
source_location: None
impact_origin: repair_recovery_cost
impact_type: repair costs
amount: around 20 000 EUR
unit: EUR

TEXT: Flood loss estimates for transportation damages and all other damages to the public sector will be in total below 500 000 EUR ”
CI_DAMAGES: ci_asset="transportation", location=None, damage="damaged"
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

"""
    ), 
    ("human", """
TEXT: {text}
CI_DAMAGES: {ci_damages}
""")
])



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




# %%

# infrastructure_impact_prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         """
# You are an infrastructure cascading-impact extraction agent.

# Your task is ONLY to identify downstream impacts affecting OTHER infrastructure assets or infrastructure-related services.

# The input contains:
# - TEXT: the original text
# - CI_DAMAGES: damage/disruption affecting an infrastructure asset or service, and its location
# Do NOT verify, modify, or reinterpret the original infrastructure extraction.
# If no infrastructure/service cascading impact is explicitly mentioned, return None.

# A cascading infrastructure impact occurs when damage or disruption to one infrastructure asset causes a further impact on another infrastructure.

# Examples:

# TEXT: The flooding of railways caused train delays.
# CI_DAMAGES: ci_asset="railways", location=None, damage="flooded"
# source_asset: railways
# source_location: None
# target_damage: delays
# target_asset: train delays

# TEXT: The bridges were so much damaged that road closures were conducted in the Ahr valley. 
# CI_DAMAGES: ci_asset="bridges", location="Ahr valley", damage="damaged"
# source_asset: bridges
# source_location: Ahr valley
# target_damage: closures
# target_asset: road closures

# TEXT: Also, the public transport was disrupted due the blocked roads.
# CI_DAMAGES: ci_asset="roads", location=None, damage="blocked"
# source_asset: roads
# source_location: None
# target_damage: disrupted
# target_asset: public transport was disrupted


# Now, its your task to extract secondary infrastructure impacts for each of the damaged infrastructure:
# TEXT: {text}
# CI_DAMAGES: {ci_damages}
# source_asset: <your_response>
# source_location  <your_response>
# target_damage: <your_response>
# target_asset: <your_response>

# """
#     ),
#     (
#         "human",
#         """
# TEXT: {text}
# CI_DAMAGES: {ci_damages}
# """
#     )])






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
""" + pp.describe_damage_categories() + """

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
CI_DAMAGES: ci_asset="railways", location=None, damage="inundation"
source_asset: railways
source_damage: flooded
source_location: None
target_asset: train service
target_damage: disruption

TEXT: The bridges were severely damaged, resulting in road closures in the Ahr valley.
CI_DAMAGES: ci_asset="bridges", location="Ahr valley", damage="structural_damage"
source_asset: bridges
source_damage: damaged
source_location: Ahr valley
target_asset: roads 
target_damage: disruption

TEXT: Public transport was disrupted because roads were blocked.
CI_DAMAGES: ci_asset="roads", location=None, damage="disruption"
source_asset: roads
source_damage: blocked
source_location: None
target_asset: public transport
target_damage: disruption

TEXT: Because the power line was destroyed, the water supply in Bad Neuenahr failed.
CI_DAMAGES: ci_asset="power line", location="Bad Neuenahr", damage="structural_damage"
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

"""
    ),
    (
        "human",
        """
TEXT: {text}
CI_DAMAGES: {ci_damages}
"""
    )
])





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

# %%
 #   Examples: "1,000 households", "10,000 people", "3 days", "up to 5 months"
 
societal_impact_prompt = ChatPromptTemplate.from_messages([
    (
        "system", """
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
CI_DAMAGES: ci_asset="water", location="Mayschloss", damage="left without"
source_asset: water
source_location: Mayschloss
impact_origin: loss_of_critical_service_or_inaccessibility
impact_type: affected people
amount: thousands
unit: people


TEXT: Similar to Germany, severe damage had been observed to the gas network. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with cost reaching more than 5 million Euros. In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021).
CI_DAMAGES: ci_asset="electricity supply", location="Netherlands", damage="loss of"
source_asset: electricity supply
source_location: Netherlands
impact_origin: loss_of_critical_service_or_inaccessibility
impact_type: affected households
amount: 1000-2000
unit: households


TEXT: After the dike break, around 2,000 residents of Grimma had to be evacuated.
CI_DAMAGES: ci_asset="dike", location="Grimma", damage="dike break(s)"
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

"""
    ),
    (
        "human",
        """
TEXT: {text}
CI_DAMAGES: {ci_damages}
"""
    )
])


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
    
    # assets_list = state["assets"]
    # damages_list = state["damages"]
    # sec_impacts_list = []

    # class SingleSocietalImpact(BaseModel):
    #     impact: Annotated[List[str], Len(min_length=1, max_length=1)] | str="NAN"
    # llm_structured = llm.with_structured_output(SingleSocietalImpact)

    # for asset, damage in zip(assets_list, damages_list):
    #     result = llm_structured.invoke([
    #             HumanMessage(
    #                 content=societal_impact_prompt.format(
    #                     text=state["text"],
    #                     assets=asset,
    #                     damages=damage
    #     ))])
    #     try:
    #         sec_impacts_list.append(result.impacts)
    #     except AttributeError:
    #         sec_impacts_list.append("NAN")
    # print("societal_impacts", sec_impacts_list)
    # return {"societal_impacts": sec_impacts_list}


# %%

environmental_impact_prompt = ChatPromptTemplate.from_messages([
    (
        "system", """
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
CI_DAMAGES: ci_asset="industrial plant", location=None, damage="inundation"
source_asset: industrial plant
source_location: None
impact_origin: environmental_impact
impact_type: pollution of environment
amount: None
unit: None


TEXT: After the dike break near Fischbeck, around 150 square kilometres of land were inundated.
CI_DAMAGES: ci_asset="dike", location="Fischbeck", damage="structural_damage"
source_asset: dike
source_location: Fischbeck
impact_origin: environmental_impact
impact_type: flooded/inundated
amount: around 150 square kilometres
unit: square kilometres


TEXT: Debris of the destroyed bridges in the Ahr valley was washed into the river.
CI_DAMAGES: ci_asset="bridges", location="Ahr valley", damage="structural_damage"
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

"""
    ),
    (
        "human",
        """
TEXT: {text}
CI_DAMAGES: {ci_damages}
"""
    )
])


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


# %%
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

# %% [markdown]
# ### BUild Graph

# %%

###########################################################################
# Graph
###########################################################################
builder = StateGraph(GraphState)

builder.add_node("is_infrastructure_case", determine_infrastructure_case)
builder.add_node("assets", extract_assets)
builder.add_node("tools", tool_node)
builder.add_node("extract_asset_information", extract_asset_information)
builder.add_node("detect_cascading_impacts", detect_cascade)
builder.add_node("economic_impacts",    extract_economic_impacts)
builder.add_node( "societal_impacts", extract_societal_impacts)
builder.add_node( "secondary_infrastructure_impacts", extract_infrastructure_impacts)
builder.add_node( "environmental_impacts", extract_environmental_impacts)

# Set arcs
builder.set_entry_point("is_infrastructure_case")
builder.add_conditional_edges("is_infrastructure_case", should_continue_is_case, ["assets", END])
builder.add_conditional_edges("assets", should_continue, ["tools", "extract_asset_information"])
builder.add_edge("tools", "extract_asset_information")
# # Cascading-impacts
builder.add_edge("extract_asset_information","detect_cascading_impacts")
builder.add_conditional_edges("detect_cascading_impacts", route_cascading_impacts)

## add memory
# memory = MemorySaver()
# graph = builder.compile(checkpointer=memory)

# init graph
graph = builder.compile()

graph

# NOTE
# NIviDiA API LIMIT: dont do parallel specialist calls, or parallel nodes
# "hitting concurrency limits. You could deliberately make the two specialist agents sequential if rate/concurrency limits become a problem.""


# %%

###########################################################################
# Example
###########################################################################

#  A few national highways were partly flooded (e.g. the A76 in both directions) or briefly closed (&lt; 3 d) because of the potential of flooding. Most likely due to relative low-flow velocities, damage to Dutch national road infrastructure was limited. Several railway sections were closed (e.g. the railway section between Maastricht and Liége) and some damage occurred to the railway infrastructure, in particular to the electronic 'track circuit' devices and saturated railway embankments (Prorail, 2021).   
# text = "In the region of Rhineland-Palatinate (Germany), most drinking water supply was restored within 2 months (Hochwasser Ahr, 2021a). However, sewage treatment plants in Altenahr, Mayschoss and Sinzig had been largely destroyed (Hochwasser Ahr, 2021b), and it is expected to take at least 1.5 years to fully repair most sewage treatment plants. Emergency sewage treatment plants have been built in the meantime (GA, 2021). In the Erft region 7 out of 31 wastewater facilities had been destroyed. Many facilities reported pollution of oil and diesel, forming layers up to 15 cm thick (Kuhn, 2021). In addition, much of the groundwater (and soil) in the flood region was mixed with oil (from destroyed residential oil tanks), chemicals such as fertilizers (from wineries and other agriculture) and chemicals from nearby industrial plants."
# text = "The most severely damaged railway line (between the villages of Spa and Pepinster) was reopened again on 3 October 2021 (Rozendaal, 2021b)."

# text2 = "There have also been some traffic jams and road closures in Madrid, such as on Lope de Vega Avenue, heading towards the city center, near Julio Cortázar Avenue due to a road subsidence. As the City Council indicated on social media, traffic has been diverted towards Colonia Santa Inés. There is also heavy traffic on the access roads from Juan XXIII Avenue to Plaza Manuel Azaña, Blas Infante Avenue, Guerrero Strachan Avenue, Velázquez Avenue, Camino Suárez, the Azucarera - Interhorce road, Santa Rosa de Lima, and Victoria In addition, there are pools of water on the MA21, near the Churriana intersection, and the intersections of Avenida Herrera Oria-Virgen de las Flores, Pasillo del Matadero Puente del Carmen, Pasillo Santa Isabel - Puente de la Aurora and Avenida Lope de Vega - Atabal have also been affected."
# text2 = """Similar to Germany, severe damage had been observed to the gas network caused by the flood event. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with repair costs reaching more than 5 million Euros. In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021)."""
text2 = """Similar to Germany, severe damage had been observed to the gas network caused by the flood event. In the villages around Liége, gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with cost reaching more than 5 million Euros. In addition, more than 10 schools were affected in Liége and surrounding. """
# text2 = "Due to a dike break on 10.06., long-distance transport routes of Deutsche Bahn AG had to be blocked in the district of Stendal, so that there were train failures and long delays."

# text2 = """In Germany, road and railway infrastructure was severely damaged as documented exemplarily in Fig. 1. Cost estimates reach up to EURO 2 billion Euro (MDR, 2021). More than 130 km of motorways were closed directly after the event, of which 50 km were still closed two months later, with an estimated repair cost of EUR 100 million (Hauser, 2021). Of the 112 bridges in the flooded 40 km of the Ahr valley (Rhineland-Palatinate), 62 bridges were destroyed, 13 were severely damaged and only 35 were in operation a month after the flood event (MDR, 2021). Over 74 km of roads, paths and bridges in the Ahr valley have been (critically) damaged. In some cases, repairs are expected to take months to years (Zeit Online, 2021). For example, major freeway sections, including parts of the A1 motorway, were closed until early 2022 (24Rhein, 2022). In addition, about 50 000 cars were damaged, causing insurance claims of some EUR450 million (ADAC, 2021). The German railway provider Deutsche Bahn expects asset damages of around EUR1.3 billion."""
# text = "There have also been some traffic jams and road closures in the capital, such as on Lope de Vega Avenue, heading towards the city center, near Julio Cortázar Avenue due to a road subsidence. As the City Council indicated on social media, traffic has been diverted towards Colonia Santa Inés. There is also heavy traffic on the access roads from Juan XXIII Avenue to Plaza Manuel Azaña, Blas Infante Avenue, Guerrero Strachan Avenue, Velázquez Avenue, Camino Suárez, the Azucarera - Interhorce road, Santa Rosa de Lima, and Victoria In addition, there are pools of water on the MA21, near the Churriana intersection, and the intersections of Avenida Herrera Oria-Virgen de las Flores, Pasillo del Matadero Puente del Carmen, Pasillo Santa Isabel - Puente de la Aurora and Avenida Lope de Vega - Atabal have also been affected."

# text = """
#     There have also been some traffic jams and road closures in Madrid
#     , such as on Lope de Vega Avenue, heading towards the city center, near Julio Cortázar Avenue due to a road subsidence. 
# As the City Council indicated on social media, traffic has been diverted towards Colonia Santa Inés. 
# There is also heavy traffic on the access roads from Juan XXIII Avenue to Plaza Manuel Azaña, Blas Infante Avenue, Guerrero Strachan Avenue, 
# Velázquez Avenue, Camino Suárez, the Azucarera - Interhorce road, Santa Rosa de Lima, and Victoria.
# In addition, there are pools of water on the MA21, near the Churriana intersection, 
# and the intersections of Avenida Herrera Oria-Virgen de las Flores, Pasillo del Matadero Puente del Carmen, 
# Pasillo Santa Isabel - Puente de la Aurora and Avenida Lope de Vega - Atabal have also been affected.
# """

# extract impacts
if RUN_EXAMPLE:
    print(text2)
    result = graph.invoke({"text": text2})
    print(result)


# %%

# belgium nl gas+ electricy supply - # a,l,d
# +5.48 min , 3.29 min, 2.30 min

# extract sec impact (add. time)
#  nach damages: 1.35min, cascIm=T -> eco impacts: reaching > 5 mill, 2.31min

# 8.30 min (when casc. impacts is captured by iterating over each asset)
# DRAWBACK: assigns secondary impacts to also ci cases, which had only direct impacts - halluciations
# Asset finalize 5 ["the gas network", "gas supply", "electricity supply", "no gas supply", "electricity supply"]
# locs None
# locs 0 5 location=['NAN', 'Chaudfontaine and Pepinster (Belgium)', 'Netherlands', 'Netherlands', 'Netherlands']
# damage m: content='System: For every infrastructure asset or service returned by ASSETS determine the damage type.\nUse for guidance the keywords from the following list to describe the damage type, but also create a new one when you cannot find it in the list: \n[affected, damaged, severely damaged, highly damaged, disrupted, blocked, destroyed, largely destroyed, heavily destroyed, partly closed, closed, outage(s), blackout, contaminated, polluted, flooded, inundated, fallen tree(s), broken tree(s), overturned tree(s), uprooted tree(s), felled tree(s), debris, debris removal, debris on, water on, dam failure(s), dyke break(s), dike break(s), breached dike(s), broken embankment(s), closure(s), derailed, impassable, traffic jam(s), traffic disruption(s), traffic congestion(s), indirectly affected, flight delay(s), train delay(s), delay(s), train failure(s), failure(s), cancellation(s), canceled, re-established, reestablished, restored, reconnected, rebuild, repaired]\n\nIf no information of a damage type is found, then return a "NAN" value.\n\n\nITEM TO EXTRACT:\n- "damage": "List the damage type of each affected infrastructure."\n\n\n\nReturn a JSON object.\n\n\nExample:\n\nci_asset:["roads","railway infrastructure"]\ndamage: ["blocked", "flooded"]\n\n\nHuman: \nTEXT: Similar to Germany, severe damage had been observed to the gas network. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with cost reaching more than 5 million Euros. In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021).\n\nASSETS: [\'the gas network\', \'gas supply\', \'electricity supply\', \'no gas supply\', \'electricity supply\']\n\n' additional_kwargs={} response_metadata={}
# damages damage=['severely damaged', 'damaged', 'outage', 'outage', 'outage']
# has_cascading_impacts has_cascading_impacts=True
# sec_impacts_list [[EconomicImpact(description='repair costs', source_asset='gas network', estimated_cost='more than 5 million Euros', currency='EUR')], [EconomicImpact(description='cost reaching more than 5 million Euros', source_asset='gas supply', estimated_cost='5 million Euros', currency='EUR'), EconomicImpact(description='loss of electricity supply for 1000-2000 households', source_asset='electricity supply', estimated_cost='N/A', currency='N/A'), EconomicImpact(description='loss of gas supply for 100-200 households', source_asset='gas supply', estimated_cost='N/A', currency='N/A')], [EconomicImpact(description='cost reaching more than 5 million Euros', source_asset='gas network', estimated_cost='5 million Euros', currency='EUR')], [EconomicImpact(description='repair costs reaching more than 5 million Euros', source_asset='gas network', estimated_cost='5 million Euros', currency='EUR')], [EconomicImpact(description='repair costs', source_asset='gas network', estimated_cost='more than 5 million Euros', currency='EUR'), EconomicImpact(description='business interruption', source_asset='electricity supply', estimated_cost='none specified', currency=None), EconomicImpact(description='loss of gas supply', source_asset='gas network', estimated_cost='none specified', currency=None)]]
# societal_impacts [[SocietalImpact(source_asset='gas network', magnitude='100–200 households lost gas supply'), SocietalImpact(source_asset='electricity supply', magnitude='1,000–2,000 households lost electricity supply'), SocietalImpact(source_asset='gas network', magnitude='Communities temporarily cut off from essential services such as heating, cooking, and refrigeration'), SocietalImpact(source_asset='electricity supply', magnitude='Communities isolated from surrounding areas for several days until power was restored'), SocietalImpact(source_asset='gas network', magnitude='Residents unable to use gas‑powered appliances, potentially affecting health and daily routines')], 'NAN', [SocietalImpact(source_asset='electricity supply', magnitude='1000-2000 households affected by outage'), SocietalImpact(source_asset='electricity supply', magnitude='1000-2000 households without electricity'), SocietalImpact(source_asset='electricity supply', magnitude='1000-2000 households unable to access essential services'), SocietalImpact(source_asset='gas supply', magnitude='100-200 households affected by outage'), SocietalImpact(source_asset='gas supply', magnitude='100-200 households without gas'), SocietalImpact(source_asset='gas supply', magnitude='100-200 households unable to access essential services')], [SocietalImpact(source_asset='gas network', magnitude='100–200 households without gas supply during the outage'), SocietalImpact(source_asset='electricity network', magnitude='1,000–2,000 households without electricity supply at the peak of the event'), SocietalImpact(source_asset='gas network', magnitude='residents unable to use gas for heating, cooking, and refrigeration'), SocietalImpact(source_asset='electricity network', magnitude='residents unable to use electric appliances, lighting, and communication devices'), SocietalImpact(source_asset='gas & electricity networks', magnitude='communities temporarily isolated from surrounding areas due to loss of power and gas'), SocietalImpact(source_asset='gas & electricity networks', magnitude='disruption of access to essential services (e.g., heating, cooking, refrigeration, communication)'), SocietalImpact(source_asset='gas & electricity networks', magnitude='potential health and safety risks for residents (e.g., hypothermia, food spoilage, lack of emergency lighting)')], [SocietalImpact(source_asset='electricity supply', magnitude='1000–2000 households lost electricity at the peak of the event'), SocietalImpact(source_asset='electricity supply', magnitude='electricity outage lasted several days'), SocietalImpact(source_asset='electricity supply', magnitude='potential disruption of essential services (e.g., refrigeration, heating, communication) for affected households'), SocietalImpact(source_asset='electricity supply', magnitude='possible isolation of communities due to lack of power for transportation and communication infrastructure')]]
# {'messages': [AIMessage(content='', additional_kwargs={'reasoning_content': 'We need to extract infrastructure assets or services damaged by a natural hazard. The text: "Similar to Germany, severe damage had been observed to the gas network. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with cost reaching more than 5 million Euros. In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021)."\n\nWe need to use get_infrastructure_entities tool to detect entities. Then decide if they are damaged by natural hazard. The text mentions severe damage to gas network. Also loss of electricity supply. So assets: gas network, electricity supply. Also households? But households are not infrastructure. So list: gas network, electricity supply. Possibly "gas supply" is service. So we include "gas network" and "electricity supply". Also "gas supply" maybe separate? But we can include "gas supply" as service. The instruction: "List of infrastructure assets or services damaged by a natural hazard." So we can list "gas network", "gas supply", "electricity supply". But we should avoid duplicates. The text: "severe damage had been observed to the gas network." So gas network is damaged. "1000-2000 households experienced a loss of electricity supply" so electricity supply damaged. "Between 100 to 200 households had no gas supply." So gas supply also damaged. So we include gas network, gas supply, electricity supply. But maybe we can combine gas network and gas supply? The instruction: "When an entry contains two or more infrastructure assets then split them in two." So if we have "gas network and gas supply" we split. So we need separate entries. So final JSON: {"ci_asset": ["gas network", "gas supply", "electricity supply"]}.\n\nWe need to call the tool.', 'reasoning': 'We need to extract infrastructure assets or services damaged by a natural hazard. The text: "Similar to Germany, severe damage had been observed to the gas network. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with cost reaching more than 5 million Euros. In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021)."\n\nWe need to use get_infrastructure_entities tool to detect entities. Then decide if they are damaged by natural hazard. The text mentions severe damage to gas network. Also loss of electricity supply. So assets: gas network, electricity supply. Also households? But households are not infrastructure. So list: gas network, electricity supply. Possibly "gas supply" is service. So we include "gas network" and "electricity supply". Also "gas supply" maybe separate? But we can include "gas supply" as service. The instruction: "List of infrastructure assets or services damaged by a natural hazard." So we can list "gas network", "gas supply", "electricity supply". But we should avoid duplicates. The text: "severe damage had been observed to the gas network." So gas network is damaged. "1000-2000 households experienced a loss of electricity supply" so electricity supply damaged. "Between 100 to 200 households had no gas supply." So gas supply also damaged. So we include gas network, gas supply, electricity supply. But maybe we can combine gas network and gas supply? The instruction: "When an entry contains two or more infrastructure assets then split them in two." So if we have "gas network and gas supply" we split. So we need separate entries. So final JSON: {"ci_asset": ["gas network", "gas supply", "electricity supply"]}.\n\nWe need to call the tool.', '_reasoning_api_fields': ['reasoning_content', 'reasoning'], 'tool_calls': [{'id': 'chatcmpl-tool-810e9bd5f333fc1c', 'type': 'function', 'function': {'name': 'get_infrastructure_entities', 'arguments': '{"text": "Similar to Germany, severe damage had been observed to the gas network. In the villages around Li\\u00e9ge, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with cost reaching more than 5 million Euros. In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021)."}'}}]}, response_metadata={'role': 'assistant', 'content': None, 'refusal': None, 'annotations': None, 'audio': None, 'function_call': None, 'tool_calls': [{'id': 'chatcmpl-tool-810e9bd5f333fc1c', 'type': 'function', 'function': {'name': 'get_infrastructure_entities', 'arguments': '{"text": "Similar to Germany, severe damage had been observed to the gas network. In the villages around Li\\u00e9ge, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with cost reaching more than 5 million Euros. In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021)."}'}}], 'reasoning': 'We need to extract infrastructure assets or services damaged by a natural hazard. The text: "Similar to Germany, severe damage had been observed to the gas network. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with cost reaching more than 5 million Euros. In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021)."\n\nWe need to use get_infrastructure_entities tool to detect entities. Then decide if they are damaged by natural hazard. The text mentions severe damage to gas network. Also loss of electricity supply. So assets: gas network, electricity supply. Also households? But households are not infrastructure. So list: gas network, electricity supply. Possibly "gas supply" is service. So we include "gas network" and "electricity supply". Also "gas supply" maybe separate? But we can include "gas supply" as service. The instruction: "List of infrastructure assets or services damaged by a natural hazard." So we can list "gas network", "gas supply", "electricity supply". But we should avoid duplicates. The text: "severe damage had been observed to the gas network." So gas network is damaged. "1000-2000 households experienced a loss of electricity supply" so electricity supply damaged. "Between 100 to 200 households had no gas supply." So gas supply also damaged. So we include gas network, gas supply, electricity supply. But maybe we can combine gas network and gas supply? The instruction: "When an entry contains two or more infrastructure assets then split them in two." So if we have "gas network and gas supply" we split. So we need separate entries. So final JSON: {"ci_asset": ["gas network", "gas supply", "electricity supply"]}.\n\nWe need to call the tool.', 'reasoning_content': 'We need to extract infrastructure assets or services damaged by a natural hazard. The text: "Similar to Germany, severe damage had been observed to the gas network. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with cost reaching more than 5 million Euros. In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021)."\n\nWe need to use get_infrastructure_entities tool to detect entities. Then decide if they are damaged by natural hazard. The text mentions severe damage to gas network. Also loss of electricity supply. So assets: gas network, electricity supply. Also households? But households are not infrastructure. So list: gas network, electricity supply. Possibly "gas supply" is service. So we include "gas network" and "electricity supply". Also "gas supply" maybe separate? But we can include "gas supply" as service. The instruction: "List of infrastructure assets or services damaged by a natural hazard." So we can list "gas network", "gas supply", "electricity supply". But we should avoid duplicates. The text: "severe damage had been observed to the gas network." So gas network is damaged. "1000-2000 households experienced a loss of electricity supply" so electricity supply damaged. "Between 100 to 200 households had no gas supply." So gas supply also damaged. So we include gas network, gas supply, electricity supply. But maybe we can combine gas network and gas supply? The instruction: "When an entry contains two or more infrastructure assets then split them in two." So if we have "gas network and gas supply" we split. So we need separate entries. So final JSON: {"ci_asset": ["gas network", "gas supply", "electricity supply"]}.\n\nWe need to call the tool.', 'token_usage': {'prompt_tokens': 422, 'total_tokens': 1015, 'completion_tokens': 593, 'prompt_tokens_details': None}, 'finish_reason': 'tool_calls', 'model_name': 'openai/gpt-oss-20b'}, id='lc_run--01a0e7fc-1442-7780-b6d0-0a2051d862f3-0', tool_calls=[{'name': 'get_infrastructure_entities', 'args': {'text': 'Similar to Germany, severe damage had been observed to the gas network. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with cost reaching more than 5 million Euros. In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021).'}, 'id': 'chatcmpl-tool-810e9bd5f333fc1c', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 422, 'output_tokens': 593, 'total_tokens': 1015}, role='assistant'), ToolMessage(content='["the gas network", "gas supply", "electricity supply", "no gas supply", "electricity supply"]', name='get_infrastructure_entities', id='cb2eb88d-391a-46b2-a6e4-379697bb8ae3', tool_call_id='chatcmpl-tool-810e9bd5f333fc1c')], 'text': 'Similar to Germany, severe damage had been observed to the gas network. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021) with cost reaching more than 5 million Euros. In the Netherlands, 1000-2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021).', 'assets': ['the gas network', 'gas supply', 'electricity supply', 'no gas supply', 'electricity supply'], 'damages': ['severely damaged', 'damaged', 'outage', 'outage', 'outage'], 'locations': ['NAN', 'Chaudfontaine and Pepinster (Belgium)', 'Netherlands', 'Netherlands', 'Netherlands'], 'has_cascading_impacts': True, 'economic_impacts': [[EconomicImpact(description='repair costs', source_asset='gas network', estimated_cost='more than 5 million Euros', currency='EUR')], [EconomicImpact(description='cost reaching more than 5 million Euros', source_asset='gas supply', estimated_cost='5 million Euros', currency='EUR'), EconomicImpact(description='loss of electricity supply for 1000-2000 households', source_asset='electricity supply', estimated_cost='N/A', currency='N/A'), EconomicImpact(description='loss of gas supply for 100-200 households', source_asset='gas supply', estimated_cost='N/A', currency='N/A')], [EconomicImpact(description='cost reaching more than 5 million Euros', source_asset='gas network', estimated_cost='5 million Euros', currency='EUR')], [EconomicImpact(description='repair costs reaching more than 5 million Euros', source_asset='gas network', estimated_cost='5 million Euros', currency='EUR')], [EconomicImpact(description='repair costs', source_asset='gas network', estimated_cost='more than 5 million Euros', currency='EUR'), EconomicImpact(description='business interruption', source_asset='electricity supply', estimated_cost='none specified', currency=None), EconomicImpact(description='loss of gas supply', source_asset='gas network', estimated_cost='none specified', currency=None)]], 'societal_impacts': [[SocietalImpact(source_asset='gas network', magnitude='100–200 households lost gas supply'), SocietalImpact(source_asset='electricity supply', magnitude='1,000–2,000 households lost electricity supply'), SocietalImpact(source_asset='gas network', magnitude='Communities temporarily cut off from essential services such as heating, cooking, and refrigeration'), SocietalImpact(source_asset='electricity supply', magnitude='Communities isolated from surrounding areas for several days until power was restored'), SocietalImpact(source_asset='gas network', magnitude='Residents unable to use gas‑powered appliances, potentially affecting health and daily routines')], 'NAN', [SocietalImpact(source_asset='electricity supply', magnitude='1000-2000 households affected by outage'), SocietalImpact(source_asset='electricity supply', magnitude='1000-2000 households without electricity'), SocietalImpact(source_asset='electricity supply', magnitude='1000-2000 households unable to access essential services'), SocietalImpact(source_asset='gas supply', magnitude='100-200 households affected by outage'), SocietalImpact(source_asset='gas supply', magnitude='100-200 households without gas'), SocietalImpact(source_asset='gas supply', magnitude='100-200 households unable to access essential services')], [SocietalImpact(source_asset='gas network', magnitude='100–200 households without gas supply during the outage'), SocietalImpact(source_asset='electricity network', magnitude='1,000–2,000 households without electricity supply at the peak of the event'), SocietalImpact(source_asset='gas network', magnitude='residents unable to use gas for heating, cooking, and refrigeration'), SocietalImpact(source_asset='electricity network', magnitude='residents unable to use electric appliances, lighting, and communication devices'), SocietalImpact(source_asset='gas & electricity networks', magnitude='communities temporarily isolated from surrounding areas due to loss of power and gas'), SocietalImpact(source_asset='gas & electricity networks', magnitude='disruption of access to essential services (e.g., heating, cooking, refrigeration, communication)'), SocietalImpact(source_asset='gas & electricity networks', magnitude='potential health and safety risks for residents (e.g., hypothermia, food spoilage, lack of emergency lighting)')], [SocietalImpact(source_asset='electricity supply', magnitude='1000–2000 households lost electricity at the peak of the event'), SocietalImpact(source_asset='electricity supply', magnitude='electricity outage lasted several days'), SocietalImpact(source_asset='electricity supply', magnitude='potential disruption of essential services (e.g., refrigeration, heating, communication) for affected households'), SocietalImpact(source_asset='electricity supply', magnitude='possible isolation of communities due to lack of power for transportation and communication infrastructure')]]} 


# %%
# result["infrastructure_cases"] = filter(lambda v: v is not None, result["infrastructure_cases"])
# result["infrastructure_cases"]


# %%

# df_resp_ci_loc_dam = pp.group_ci_types(df_resp, "infrastructure_type", "infrastructure_group", ci_patterns)
# df_resp_ci_loc_dam.dropna(subset=["infrastructure_group"], inplace=True)
# df_resp_ci_loc_dam
# 
# # df_resp["infrastructure_group"] = None            
# df_resp_ci_loc_dam = pp.group_ci_types(df_resp, "infrastructure_type", "infrastructure_group", ci_patterns)
# df_resp_ci_loc_dam.dropna(subset=["infrastructure_group"], inplace=True)
# df_resp_ci_loc_dam

# %% [markdown]
# ##### Postprocess response

# %%
import pandas as pd
import re


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
    infrastructure_cases = data.get("infrastructure_cases", [])  or []
    economic_impacts = data.get("economic_impacts", []) or []
    societal_impacts = data.get("societal_impacts", []) or []
    infrastructure_impacts = data.get("secondary_infrastructure_impacts", [])  or []
    environmental_impacts = data.get("environmental_impacts", []) or []

    # Create base dataframe
    df = pd.DataFrame([case.model_dump() for case in infrastructure_cases.cases])

    if df.empty:
        return df

    # Normalize asset names for matching
    df["_asset_normalized"] = (
        df["ci_asset"]
        .apply(normalize_asset)
    )

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
    # df["secondary_infrastructure_impact_type"] = ""

    # ECONOMIC IMPACTS
    for impact in economic_impacts:

        source_asset = normalize_asset(
            impact.source_asset
        )

        if not source_asset:
            continue

        matches = (
            df["_asset_normalized"]
            == source_asset
        )

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

            df.at[idx, "economic_impact_type"] = (
                append_value(
                    df.at[idx, "economic_impact_type"],
                    impact_type
            ))
            df.at[idx, "economic_impact_origin"] = (
                append_value(
                    df.at[idx, "economic_impact_origin"],
                    impact_origin
            ))
            df.at[idx, "economic_impact_amount"] = (
                append_value(
                    df.at[idx, "economic_impact_amount"],
                    impact_amount
            ))
            df.at[idx, "economic_impact_unit"] = (
                append_value(
                    df.at[idx, "economic_impact_unit"],
                    impact_unit
            ))
            

    # SOCIETAL IMPACTS
    for impact in societal_impacts:

        source_asset = normalize_asset(
            impact.source_asset
        )

        if not source_asset:
            continue

        impact_origin = impact.impact_origin
        impact_type = impact.impact_type
        impact_amount = impact.amount
        impact_unit = impact.unit

        matches = (
            df["_asset_normalized"]
            == source_asset
        )

        for idx in df.index[matches]:

            df.at[idx, "societal_impact_origin"] = (
                append_value(
                    df.at[idx, "societal_impact_origin"],
                    impact_origin
            ))
            df.at[idx, "societal_impact_type"] = (
                append_value(
                    df.at[idx, "societal_impact_type"],
                    impact_type
                )
            )
            df.at[idx, "societal_impact_amount"] = (
                append_value(
                    df.at[idx, "societal_impact_amount"],
                    impact_amount
            ))
            df.at[idx, "societal_impact_unit"] = (
                append_value(
                    df.at[idx, "societal_impact_unit"],
                    impact_unit
            ))

    # ENVIRONMENTAL IMPACTS
    for impact in environmental_impacts:

        source_asset = normalize_asset(
            impact.source_asset
        )

        if not source_asset:
            continue

        matches = (
            df["_asset_normalized"]
            == source_asset
        )

        for idx in df.index[matches]:

            df.at[idx, "environmental_impact_origin"] = (
                append_value(
                    df.at[idx, "environmental_impact_origin"],
                    impact.impact_origin
            ))
            df.at[idx, "environmental_impact_type"] = (
                append_value(
                    df.at[idx, "environmental_impact_type"],
                    impact.impact_type
            ))
            df.at[idx, "environmental_impact_amount"] = (
                append_value(
                    df.at[idx, "environmental_impact_amount"],
                    impact.amount
            ))
            df.at[idx, "environmental_impact_unit"] = (
                append_value(
                    df.at[idx, "environmental_impact_unit"],
                    impact.unit
            ))

    # SECONDARY INFRASTRUCTURE IMPACTS
    for impact in infrastructure_impacts:

        source_asset = normalize_asset(
            impact.source_asset
        )

        if not source_asset:
            continue

        impact_target_asset = impact.target_asset
        
    
        matches = (
            df["_asset_normalized"]
            == source_asset
        )

        for idx in df.index[matches]:

            # df.at[idx, "secondary_infrastructure_asset"] = (
            #     append_value(
            #         df.at[idx, "secondary_infrastructure_asset"],
            #         impact_secondary_asset
            # ))
            # df.at[idx, "secondary_infrastructure_impact_type"] = (
            #     append_value(
            #         df.at[idx, "secondary_infrastructure_impact_type"],
            #         impact_secondary_type
            # ))
            df.at[idx, "secondary_infrastructure_asset"] = (
                append_value(
                    df.at[idx, "secondary_infrastructure_asset"],
                    impact_target_asset
            ))
    # Remove helper column
    df = df.drop(
        columns="_asset_normalized"
    )

    return df


# # %%
# print(text2)
# df_resp = impacts_to_dataframe(r).replace("; ",", ", regex=True)
# df_resp["has_cascading_impacts"] = r.get("has_cascading_impacts", []).has_cascading_impacts
# df_resp

# %%


# %%
#     df_resp = pd.DataFrame(dict(
#         (k, result[k]) for k in ['assets', "damages", 'locations'] if k in result)
#     )

# # # disentangle rows which contain multiple locations or CIs
# df_resp = pp.split_text_into_multiple_rows(df_resp, "location",  split_at = " and ")
# df_resp = pp.split_text_into_multiple_rows(df_resp, "location",  split_at = " or ")
# df_resp = pp.split_text_into_multiple_rows(df_resp, "location",  split_at = "to ")
# df_resp = pp.split_text_into_multiple_rows(df_resp, "location",  split_at = ", ")  # Germany, Neterlands, Belgium
# df_resp = pp.split_text_into_multiple_rows(df_resp, "location",  split_at = "; ")
# df_resp

# %%
# ## condense cascading infos

# df_resp_fig = df_resp

# # df_resp_fig.assets = # make as ci_group

# df_resp_fig.economic_impacts = df_resp_fig.economic_impacts.astype(bool)
# df_resp_fig['economic_impacts'] = df_resp_fig['economic_impacts'].map({True:"economic_impacts"})
# df_resp_fig

# df_resp_fig.societal_impacts = df_resp_fig.societal_impacts.astype(bool)
# df_resp_fig['societal_impacts'] = df_resp_fig['societal_impacts'].map({True:"societal_impacts"})
# df_resp_fig

# df_resp = pp.split_ci_entities(df_resp, ["ci_asset"]) # e.g "road and rail infrastructure" -> "road infrastructure", "rail infrastructure"
# df_resp = pp.split_text_into_multiple_rows(df_resp, "ci_asset",  split_at = "; ") 

# %% [markdown]
# #### GeoLLM

# %%

if use_geollama:

    import geopandas as gpd
    from geollama.geollama.main import GeoLlama
    from geollama.geollama.model import TopoModel, RAGModel 
    from shapely.geometry import shape

    topo_model = TopoModel(
        model_name='JoeShingleton/GeoLlama-3.2-3b-toponym',
        # model_name='JoeShingleton/GeoLlama_7b_toponym', 
        prompt_path='../geollama/data/prompt_templates/prompt_template.txt',
        instruct_path='../geollama/data/prompt_templates/topo_instruction.txt',
        input_path=None,
        config_path='../geollama/data/config_files/model_config.json'
    )

    rag_model = RAGModel(
        model_name='JoeShingleton/GeoLlama-3.2-3b-RAG',
        # model_name='JoeShingleton/GeoLlama_7b_RAG',
        prompt_path='../geollama/data/prompt_templates/prompt_template.txt',
        instruct_path='../geollama/data/prompt_templates/rag_instruction.txt',
        input_path='../geollama/data/prompt_templates/rag_input.txt',
        config_path='../geollama/data/config_files/model_config.json')

    geo_llama = GeoLlama(
        topo_model = topo_model, 
        rag_model = rag_model, 
        translate_model=None
    )

    resp = geo_llama.geoparse(text2)

    df_geollama_resp = pd.DataFrame()

    resp
    # df_geollama_response = pd.concat([df_geollama_response, df_geollama_resp], ignore_index=True)


    # try:
    #     locations_per_chunk = df_geollama_resp["name"].to_list()
    #     geometries_per_chunk = df_geollama_resp["geometry"].to_list()
    #     metainfo_per_chunk = df_geollama_resp["metainfo"].to_list()
    #     # lons_per_chunk = df_geollama_resp["longitude"].to_list()
    #     RAGestimated_per_chunk = df_geollama_resp["RAG_estimated"].to_list()
    # except:
    #     print("No locations extracted by geollama for this chunk. Going to next chunk\n", resp[0:])
    #     continue

    # # sanity check that only cases with location infos are considered for further processing      
    # if len(locations_per_chunk) == 0:
    #     print("geollama did not find any potential locations found for this chunk. Continue with next chunk")
    #     continue


    # print("Enriching LLm 1 response with GeoLlama locations")
    # df_resp["coord_potential_locations"] = str(dict(zip(locations_per_chunk, zip(geometries_per_chunk, RAGestimated_per_chunk, metainfo_per_chunk)))) # INTERIM for verification of lat, lon 

    



# %%


# %% [markdown]
# #### LLM-as-a-judge

# %%
###########################################################################
# LLM-as-a-judge (step 3)
# check each extracted record of a chunk on its plausibility, based ONLY on the chunk text
# the judge can accept ("correct") or adapt ("adapted") a record, or mark it as "false" - records are never removed
# the judge uses the same field definitions, keyword lists, location rules and NER entities as LLM 1
###########################################################################
import importlib
import openai
import src.llm_judge as lj
importlib.reload(lj)  # pick up changes in src/llm_judge.py without restarting the kernel


# record column -> field of the extraction schema (pydantic class) whose definition is given to the judge
judge_field_schema = {
    "infrastructure_type": (InfrastructureCase, "ci_asset"),
    "infrastructure_amount": (InfrastructureCase, "ci_asset_amount"),
    "infrastructure_name": (InfrastructureCase, "ci_asset_name"),
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
# record fields the judge checks and may adapt (metadata like chunk_text, citation_id, coords are not judged)
judge_fields = list(judge_field_schema)

# derived columns (CI grouping), never shown to or adapted by the judge
judge_excluded_fields = ["infrastructure_group", "infrastructure_sector", "secondary_infrastructure_group", "secondary_infrastructure_sector"]
assert not set(judge_fields) & set(judge_excluded_fields)

# fields with a fixed keyword list in the schema (Literal), e.g. damage, economic_impact_origin -> judge may only use these keywords
judge_allowed_values = {
    column: lj.literal_values(model.model_fields[name].annotation)
    for column, (model, name) in judge_field_schema.items()
    if lj.literal_values(model.model_fields[name].annotation)
}

# location rules: the same "IMPORTANT" points about locations as in the prompt of LLM 1 (asset_information_prompt)
location_rules = [
    line.strip()
    for line in asset_information_prompt.messages[0].prompt.template.split("Example Input")[0].split("\n")
    if line.strip().startswith("-") and "location" in line.lower() and "damage" not in line.lower()
]
assert location_rules, "no location rules found in asset_information_prompt"

# additional rules per field, on top of the field definition of the schema
judge_extra_rules = {
    "infrastructure_type": [
        "Use the INFRASTRUCTURE ENTITIES DETECTED BY NER as guidance and keep their phrasing, as LLM 1 had to.",
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
        "Use the INFRASTRUCTURE ENTITIES DETECTED BY NER as guidance and keep their phrasing, as LLM 1 had to. Never invent infrastructure.",
    ],
}


def build_judge_field_rules() -> str:
    """Definition, allowed keywords and extra rules for each judged field, as text for the judge prompt."""
    blocks = []
    for column, (model, name) in judge_field_schema.items():
        lines = [f'"{column}":', f"- Definition: {model.model_fields[name].description}"]
        if column in judge_allowed_values:
            lines.append(f"- Allowed values (or null): {', '.join(judge_allowed_values[column])}")
        else:
            lines.append("- Free text: the examples in the definition are NOT a closed list of allowed values. Keep the wording of the TEXT.")
        lines += [rule if rule.startswith("-") else f"- {rule}" for rule in judge_extra_rules.get(column, [])]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


judge_field_rules = build_judge_field_rules()


judge_prompt = ChatPromptTemplate.from_messages([
    ("system",
"""You are a strict fact-checker (LLM-as-a-judge) for information about impacts of natural hazards on critical infrastructure.

You get a TEXT, the INFRASTRUCTURE ENTITIES DETECTED BY NER in this TEXT, and a list of RECORDS that another model (LLM 1) extracted from this TEXT.
Check each record against the TEXT and the FIELD RULES below. LLM 1 had to follow the same FIELD RULES.

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
"""),
    ("human", "TEXT:\n{text}\n\nINFRASTRUCTURE ENTITIES DETECTED BY NER: {ner_entities}\n\nRECORDS:\n{records}"),
])


# after this number of consecutive failed calls the judge is skipped for the rest of the run (records get "not_judged"),
# e.g. when the model endpoint does not respond, instead of waiting for the timeout in every chunk
judge_max_consecutive_failures = 3
judge_consecutive_failures = 0
judge_last_error = None  # shown when the judge is skipped, as the per-attempt messages may be far up in the output

# max. number of records judged in one LLM call; larger chunks are judged in several calls
# (keeps reasoning + answer of the judge within max_tokens of llm_judge)
judge_batch_size = 5 # for a100: less request in parallel processed, default: 10 


def call_judge(df_batch: pd.DataFrame, chunk_text: str, ner_entities: list[str], llm, max_retries: int = 1) -> list[dict] | None:
    """Ask the LLM to judge the records of df_batch. Returns the parsed judgements, or None if all attempts failed."""
    global judge_consecutive_failures, judge_last_error

    messages = judge_prompt.format_messages(
        text=chunk_text,
        ner_entities=json.dumps(ner_entities, ensure_ascii=False),
        records=lj.records_to_judge_payload(df_batch, judge_fields),
        field_rules=judge_field_rules,
    )
    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(messages)
            finish_reason = response.response_metadata.get("finish_reason")
            if not response.content.strip() and finish_reason == "length":
                # reasoning models (e.g. gemma-4 in thinking mode) can use up all tokens for reasoning before writing the answer
                usage = response.response_metadata.get("token_usage", {}) or {}
                raise ValueError(
                    f"answer cut off before any JSON was written (finish_reason=length, completion_tokens={usage.get('completion_tokens')}). "
                    "Increase max_tokens of llm_judge or decrease judge_batch_size."
                )
            judgements = lj.parse_judge_response(response.content)
            judge_consecutive_failures = 0
            return judgements
        except Exception as e:
            judge_last_error = f"{type(e).__name__}: {e}"
            print(f"! LLM-as-a-judge failed (attempt {attempt + 1}/{max_retries + 1}): {judge_last_error}")
            # model not found / wrong api key: retrying does not help
            if isinstance(e, (openai.NotFoundError, openai.AuthenticationError, openai.PermissionDeniedError)):
                break
            if attempt < max_retries:
                time.sleep(30)

    judge_consecutive_failures += 1
    return None


def judge_records(df_records: pd.DataFrame, chunk_text: str, llm, max_retries: int = 1) -> pd.DataFrame:
    """Let the LLM judge each record of df_records against chunk_text (in batches of judge_batch_size records).
    Returns df_records (same rows) with adapted values and the judge columns, see lj.JUDGE_COLUMNS."""

    # same NER tool as for LLM 1, as guidance for infrastructure_type and secondary_infrastructure_asset
    ner_entities = get_infrastructure_entities.invoke({"text": chunk_text}) or []

    df_batches_judged = []
    for batch_start in range(0, max(len(df_records), 1), judge_batch_size):
        df_batch = df_records.iloc[batch_start:batch_start + judge_batch_size]

        judgements = []
        if judge_consecutive_failures >= judge_max_consecutive_failures:
            print(f"! LLM-as-a-judge skipped: failed {judge_consecutive_failures} times in a row. Re-run the LLM-as-a-judge cell to reset.\n  Last error: {judge_last_error}")
        elif len(df_batch):
            judgements = call_judge(df_batch, chunk_text, ner_entities, llm, max_retries=max_retries) or []

        # record_ids in the judgements refer to the positions within this batch
        df_batches_judged.append(lj.apply_judgements(
            df_batch, judgements, judge_fields,
            chunk_text=chunk_text,
            allowed_values=judge_allowed_values,
            # ignore changes that only add/remove articles, quantifiers or numbers, e.g. "bridges" -> "Two bridges"
            normalize_values={"infrastructure_type": lj.strip_determiners, "secondary_infrastructure_asset": lj.strip_determiners},
            protect_values_in_text=True  # NOTE: enable to set field to false even when its value appears in the text
        ))

    df_judged = pd.concat(df_batches_judged)

    return df_judged


# %%


# %% [markdown]
# ## Alternative LLMjudge models

# %%
# smoke test of the vLLM servers (LLM 1 and LLM-as-a-judge): both should list their served model
print(" SMOKE TEST of the vLLM servers ")
print(llm.root_client.models.list())
print(llm.invoke("What is 17 * 23?").content)
smoke_prompt = ChatPromptTemplate.from_messages([
    ("system","Extract the damaged infrastructure assets from the TEXT using the tool."), 
    ("human", "TEXT: {text}")])
messages = smoke_prompt.format_messages(text="Flood damage to railway track worth tens of millions of euros")
print(llm_tool.invoke(messages).tool_calls)
print(structured_llm(InfrastructureCases).invoke(messages))

print(llm_judge.root_client.models.list())
print(llm_judge.invoke("What is 17 * 23?").content)


# %%

# time_judge = time.time()
# df_resp_judged = pd.DataFrame()
# df = judge_records(df_resp, text2, llm_judge)
# print(lj.format_judge_report(df, verdicts="adapted"))
# df_resp_judged = pd.concat([df_resp_judged, df], ignore_index=True)
# print(f"calling LLM-as-a-judge took {np.round((time.time() - time_judge) / 60, 1)} min")
# df_resp_judged


# %% [markdown]
# ## Run LangGraph

# %%
# from scalene import scalene_profiler
# scalene_profile_2 = scalene_profiler
  ## remove "NAN"
def convert_nan(series: pd.Series) -> pd.Series:
    """ convert representations of "NAN" to np.nan """
    # TODO use regex instead of ["NAN", "NaN", "nan"] by setting all possible representations of nan (e.g. "Nan") to lowercase 
    series = series.replace(["NAN", "NaN", "nan", "none","None"], np.nan)

    return series


# %%
# %%scalene --reduced-profile  
# # # Turn profiling on
# scalene_profiler.start()



# Settings
time0 = time.time()

# clean up before applying CUDA
gc.collect()
torch.cuda.empty_cache() 
print(torch.cuda.memory_reserved() / 1e9)
torch.no_grad()


# load tokenizer
embed_model =  "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(embed_model),
    max_tokens=256, # max tokens for MiniLM-l6-v2, set here explicitly
    # standardize input sizes of chunks for Llama models
    padding=True, # add zero as extra tokens to too short sequences so that they have the same length as other chunks
    truncation=True, # truncates too long sequences (> max_tokens). If False, they will be split into multiple chunks
)

## init chunker - based on hierachical chunker but also considers max token leng, merge smaller chunks, except when at end of paragraph (merge_peers=True)
chunker = HybridChunker(
    tokenizer=tokenizer,
        # max_tokens=256, # max tokens for MiniLM-l6-v2, set here explicitly
        # chunk_overlap=0, # no overlap between chunks, as we use merge_peers to merge smaller chunks and avoid splits in sentences
    split_by_sentence=True, # split by sentence first before merging smaller chunks, to avoid splits in sentence middle
    merge_peers=True,  # optional, defaults to True
)

# init NER model pipe
nlp = NERModel().nlp_model



# load NER patterns for CI types and their subgroups (needed for cleaning LLm response - STEP 1)
ci_loc_patterns = pd.read_json("./ner_patterns.jsonl/patterns", lines=True)



gc.collect()
torch.cuda.empty_cache() 
torch.no_grad()
print(torch.cuda.memory_reserved() / 1e9)

# print(os.environ["CUDA_VISIBLE_DEVICES"])
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# print(os.environ["CUDA_VISIBLE_DEVICES"])

# for Doc parsing and cleaning
md_converter = DocumentConverter(allowed_formats=[InputFormat.MD])


# CI-GEO pairs
geolocs_cache = geonamescache.GeonamesCache()
countries = geolocs_cache.get_countries()
ci_geo_countries = [*u.gen_dict_extract(countries, 'name')]



## init outputs
df_resp_step1_all = pd.DataFrame()
df_resp_step2_all = pd.DataFrame()
responses_error_list = [] 
df_ci_cases_not_grouped = pd.DataFrame()
# df_geollama_response = pd.DataFrame()


if test_mode:
    search_path = docs_list_sample
    print("Test mode is ON. Using only a small sample of documents for testing.")
else:
    search_path = glob(str(Path(PARSED_TEXT_DIR, "*cleaned.jsonl")))

os.makedirs("../../interim_results_debug", exist_ok=True)


## Start CI impact extraction
for file_no, filename in enumerate(search_path):

    time1 = time.time()

    src_language_nonengl = None 

    no_documents = len(search_path)
    
    # load input docs
    # org_filepath = Path(filename)
    org_filepath = Path(os.path.join(DOCS_DIR, Path(filename)))
    filename_stem = org_filepath.stem
    md_filename = filename_stem + ".md"
    md_filepath = os.path.join("./", PARSED_TEXT_DIR, Path(md_filename))
    cleaned_md_filepath = md_filepath.replace(".md", "_cleaned.md")


    print(f"\n\n ######## -------- Processing document [{file_no+1}/{no_documents}]: {org_filepath.name} -------- ######## \n")

    ## extract authors, publication year and title 
    author, year, title = extract_citation_info(filename_stem)
    citation = f"{author} {year}".replace("  ", " ").strip()
    title = title.replace(" - ", "").replace("_cleaned", "").strip()


    df_resp_step1_end = pd.DataFrame()   
    df_resp_step2 = pd.DataFrame(
        # columns=[
        #     "citation_id",
        #     "chunk_id",
        #     "infrastructure_type",
        #     "infrastructure_group",
        #     "location",
        #     "location_type",
        #     "damage",
        #     "damage_value",
        #     # "ci_entity",
        #     # "geo_entity",
        #     # "geo_2_entity",
        #     "coords",
        #     # "coord_potential_locations",
        #     "chunk_text",
        #     # "verified_chunk_relevant",
        #     "verified_ci_loc",
        # ]
    )
    df_resp_judged = pd.DataFrame()  # records of all chunks incl. verdicts of LLM-as-a-judge (step 3)

    
    print(f"\n ##### ------- Parsing & Cleaning HTML document -----------########")
    # TODO 
    # incl HTLM parsing into pipeline
    
    
    print(f"\n ##### ------- Parsing & Cleaning PDF document -----------########")


    time_cleaning = time.time()

    if os.path.exists(cleaned_md_filepath):
        print("Cleaned markdown file exists, loading it")
        doclingdoc = md_converter.convert(cleaned_md_filepath).document
    else:
        print(f"No cleaned markdown file for '{filename}' ({cleaned_md_filepath}). Skipping document.")
        continue

    print(f"Document cleaning took, {np.round((time.time() - time_cleaning) / 60, 1)} minutes")



    print("Chunking document...")
    chunk_iter = chunker.chunk(dl_doc=doclingdoc)  # NOTE cannot use chunker when re-created Docl.Document with cleaned text and old DoclingObject (from converter)

    doc = list(chunk_iter)
    # print(len(chunks), "chunks created with chunker.chunk(dl_doc=doclingdoc)")
    # print(len(texts_clean), "cleaned text items in doclingdoc.texts")


    for chunk_no, chunk in enumerate(doc):

        # scalene_profile_2.start()

        df_resp_ci_loc_dam = pd.DataFrame(
            # columns=[
            #     "citation_id",
            #     "chunk_id",
            #     "infrastructure_type",
            #     "location",
            #     "location_type",
            #     "damage",
            #     "damage_value",
            #     "chunk_text",
            #     "verified_chunk_relevant",
            # ]
        )
        
        df_ci_geo_chunk = pd.DataFrame(
            columns=["citation_id", "ci_entity", "geo_entity", "geo_2_entity","chunk_text"]
        )

    
        print(f"\nProcessing chunk no. {chunk_no+1} / {len(doc)} of document: {org_filepath.name}")
        print("Chunk text: ", chunk.text)


        ## add punctuations back for e.g. and i.e. (for better reading in chunks by LLm-WF)
        chunk.text = re.sub(r"eg\s+", "e.g. ", chunk.text)
        chunk.text = re.sub(r"ie\s+", "i.e. ", chunk.text)

        ## FIXME TODO fix wrongly removed hyphens or overseen by OCR models
        chunk.text = re.sub("SachsenAnhalt", "Sachsen-Anhalt", chunk.text)
        chunk.text = re.sub("ElbeHavel", "Elbe-Havel", chunk.text)
        chunk.text = re.sub("solidwaste", "solid waste", chunk.text)
        chunk.text = re.sub("spatialtemporal", "spatial-temporal", chunk.text)
        
        
        
       

        print("Getting geolocations of CI assets ")       
        ## get most likely geolocation for each CI entity based on distance between tokens


        ## cleaning:
        # remove common errorneous words (damage, training) 
        # #which are considered as CI keywords ("dam" - "damage.*", "train"- "training") as NER patterns not recognize stop-char "\b" for any reason
        regex = re.compile(r"\b[Dd]amage\w*|\b[Tt]raining\w*")
        text_clean = re.sub(regex, "", chunk.text).replace("  ", " ")
            
        # get entities
        nlp_chunk = nlp(text_clean)
        all_ents = [ent for ent in nlp_chunk.ents]
        ci_type_ents = [ent for ent in nlp_chunk.ents if ent.label_ in ["CI_TYPE"]]#, "FAC"]]


        ci_type_ents = [str(i) for i in ci_type_ents if not regex.match(str(i))]
        
        # check if chunk contains CI_TYPE entities
        # NOTE keep ci existance checkign also for BASELINE approach
        if len(ci_type_ents) > 0:
            pass
        else:
            print("No CI_TYPE entities found in this chunk. Going to next chunk")
            continue
        
        
        ## write Ci entities to NER DF
        df_ci_geo_chunk["ci_entity"] = ci_type_ents
        

        
        # ## post-process of DF CI-GEO pairs for each chunk
        # unique_ci_geo_pairs = df_ci_geo_chunk.drop_duplicates(
        #     subset=["citation_id", "ci_entity", "geo_entity", "geo_2_entity","chunk_text"])
        # print("number of duplicates to remove:", len(df_ci_geo_chunk) - len(unique_ci_geo_pairs))

        df_ci_geo_chunk = df_ci_geo_chunk.drop_duplicates(
            subset=["citation_id", "ci_entity", "geo_entity", "geo_2_entity", "chunk_text"]
        )
        ## remove common errorneous CI keywords ("dam" - "damage.*", "train"- "training") as NER patterns not recognize stop-char "\b" for any reason
        df_ci_geo_chunk = df_ci_geo_chunk[~df_ci_geo_chunk["ci_entity"].str.contains(r".*damage.*|.*training.*", regex=True)]
      

        # print(os.environ["CUDA_VISIBLE_DEVICES"])
        # # os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
        # print(os.environ["CUDA_VISIBLE_DEVICES"])

        print("\nSTEP 1")

        ## apply decoder on each chunk in document
        ## TODO replace iteration by loading entire document and use recursive chunking from langchain
        time2 = time.time()


        # clean up before applying CUDA
        gc.collect()
        torch.cuda.empty_cache() 
        torch.no_grad()
        # print(torch.cuda.memory_reserved() / 1e9)

     
        context = [{"text": chunk.text,},]        
        graph = builder.compile()
        
    
        ## apply LLM (is_case, ci)
        time_llm1 = time.time()

        ########### chain without tool but CI NER in prompt ###############
        try:
            result = graph.invoke({"text": chunk.text})# ["extracted_cases"]
            print(result)
            
            # CI damage exists/not exists
            is_damage = result.get("is_infrastructure_case", [])
            print("is_damage", is_damage)
            if is_damage == False:
                print("! No identified CI damage, going to next chunk" )
                continue
                       
            # process to DF only when damage Ci was found
            df_resp = impacts_to_dataframe(result).replace("; ", ", ", regex=True)
            df_resp["has_cascading_impacts"] = result.get("has_cascading_impacts", []).has_cascading_impacts
            df_resp["assets_org"] = result.get("assets_org", []).cases  # backup to see if assets/loc/dam where modified in CascaImpactExtraction
            df_resp.rename(columns={"ci_asset": "infrastructure_type", "ci_asset_amount": "infrastructure_amount", "ci_asset_name": "infrastructure_name"},  inplace=True)

            # df_resp = pd.DataFrame({"infrastructure_type":k.ci_asset, "location":k.location, "damage":k.damage} for k in result["infrastructure_cases"].cases)

        # gpt-oss reasoned until max_tokens: retrying is deterministic (temperature=0, seed), so skip the chunk
        except (LengthLimitError, openai.LengthFinishReasonError) as e:
            print(f"! Token limit reached, going to next chunk: {e}")
            responses_error_list.append({
                "citation_id": citation,
                "chunk_id": chunk_no,
                "response": None,
                "error": f"{type(e).__name__}: {e}"
            })
            continue

        except Exception  as e:
            print(f"! ERROR {e}. Sleeping for 1 min and then try again")
            time.sleep(60)

            try: 
                
                result = graph.invoke({"text": chunk.text}) # ["extracted_cases"]
                print(result)
                
                # CI damage exists/not exists
                is_damage = result.get("is_infrastructure_case", [])
                print("is_damage", is_damage)
                if is_damage == False:
                    print("! No identified CI damage, going to next chunk" )
                    continue

                # process to DF only when damage Ci was found
                df_resp = impacts_to_dataframe(result).replace("; ", ", ", regex=True)    
                df_resp["has_cascading_impacts"] = result.get("has_cascading_impacts", []).has_cascading_impacts          
                df_resp["assets_org"] = result.get("assets_org", []).cases  # backup to see if assets/loc/dam where modified in CascaImpactExtraction
                df_resp.rename(columns={"ci_asset": "infrastructure_type", "ci_asset_amount": "infrastructure_amount", "ci_asset_name": "infrastructure_name"},  inplace=True)
            
            except Exception as e:
                print(f"! Could not call model going to next chunk: {e}")
                continue

   
        ## postprocess response
        try:
            try:
           
                pass

                # print("disentangle rows which contain multiple locations and CI")
                # df_resp = pp.split_text_into_multiple_rows(df_resp, "location",  split_at = " and ")
                # df_resp = pp.split_text_into_multiple_rows(df_resp, "location",  split_at = " or ")
                # df_resp = pp.split_text_into_multiple_rows(df_resp, "location",  split_at = "\bto ")
                # df_resp = pp.split_text_into_multiple_rows(df_resp, "location",  split_at = ", ")  # Germany, Neterlands, Belgium
                # df_resp = pp.split_text_into_multiple_rows(df_resp, "location",  split_at = "; ") 
                # df_resp = pp.split_text_into_multiple_rows(df_resp, "infrastructure_type",  split_at = "; ") 

                # NOTE: not split CI by "and" as then also economic costs are splitted into two
                # df_resp = pp.split_ci_entities(df_resp, ["infrastructure_type"]) # e.g "road and rail infrastructure" -> "road infrastructure", "rail infrastructure"

                # # check if it is impact to ci
                # if df_resp["verified_chunk_relevant"].all() == False:
                #     continue

                    #df_resp = pp.postprocess_response(response_until_ci[0])
            except Exception as e:
            # except (IndexError, ValueError) as e:
                print(f"Error in postprocessing respone {e}", df_resp)
                continue
                    # df_resp = pp.postprocess_response(response_until_ci)
                
                # # check if df is empty
                # if "infrastructure_type" not in df_resp.columns:
                #     continue


            # enrich LLM response (per chunk)
            df_resp["citation_id"] = citation 
            df_resp["chunk_id"] = chunk_no  # add chunk id as identifier
            df_resp["chunk_text"] =  context[0]["text"]  # add (translated) chunk text for tracing back LLM response

            if not len(df_resp):
                print("LLM response is empty. Continue with next chunk.\n Response was:", result)
                continue

                # # remove infrastructure_types which not really occur in chunk
                ## NOTE: would also remove "metro" when text is about certain subway stations
                # for case_no in df_resp.index:
                #     smlrty = fuzz.partial_ratio(df_resp["infrastructure_type"][case_no], df_resp["chunk_text"][case_no])

                #     # remove CI case when it is not in chunk
                #     if smlrty < 70:   #  eg. 73: railway infrastructure <-> This is some text about railways that are infrastructure with damages
                #         print(f"! CI type {df_resp["infrastructure_type"]} (partial ratio) does not occur in chunk text. Removing case")
                #         df_resp = df_resp[df_resp["infrastructure_type"] != df_resp["infrastructure_type"][case_no]]
                #         continue   

        except Exception as e: # (IndexError, ValueError, KeyError) as e:
            print(f"Cannot add response [until ci]: {e},\nFaulty response (before postprocessing):", result)
            # faulty response: e.g.  .., "location": "V" on satellite and online on radar"}, { ...}, {}
            responses_error_list.append({
                "citation_id": citation,
                "chunk_id": chunk_no,
                "response": result,
                "error": str(e)
            })
            print("Continue with next chunk\n")
            continue

        # # ELSE get response_ci as df[column]
        # df_resp_after_ci = pd.concat([df_resp_after_ci, df_resp], ignore_index=True)
        

        # df_resp_ci_loc_dam = df_resp_ci_loc_dam.explode("location")
        # try:
        #     # in case it is not already converted to list
        #     df_resp_ci_loc_dam["damage"] = df_resp_ci_loc_dam["damage"].apply(ast.literal_eval) #convert to list type
        # except Exception as e: 
        #     pass
        # df_resp_ci_loc_dam = df_resp_ci_loc_dam.explode("damage")

  
        df_resp_ci_loc_dam = df_resp
        # df_resp_ci_loc_dam = df_resp_ci_loc_dam.rename(columns={"assets": "infrastructure_type", "locations": "location", "damages": "damage"})
        

        ## group Ci types into subgroups, 
        if df_resp_ci_loc_dam["infrastructure_type"].isna().all() or df_resp_ci_loc_dam["location"].isna().all():
            print("No infrastructure types or locations extracted for this chunk, skip grouping into subgroups. Go to next chunk")
            print(df_resp_ci_loc_dam[["infrastructure_type", "location", "chunk_text"]])
            continue
        
        df_resp_ci_loc_dam["infrastructure_type"] = df_resp_ci_loc_dam["infrastructure_type"].replace(r"[\(\),]", "", regex=True)   # clean up CI from brackets and commas
        
        # split multiple CI entries  ("road and rail network" -> road network, rail network)
        # df_resp_ci_loc_dam = pp.split_ci_entities(df_resp_ci_loc_dam, columns=["infrastructure_type"])
        
        # CI GROUPS
        if "infrastructure_group" in df_resp.columns:
            df_resp_ci_loc_dam = pp.group_ci_types(df_resp, "infrastructure_type", "infrastructure_group", ci_patterns)
        else:
            df_resp_ci_loc_dam["infrastructure_group"] = None            
            df_resp_ci_loc_dam = pp.group_ci_types(df_resp_ci_loc_dam, "infrastructure_type", "infrastructure_group", ci_patterns)
        
        ## store cases which could not be grouped
        df_ci_cases_not_grouped = pd.concat([df_ci_cases_not_grouped, df_resp_ci_loc_dam[df_resp_ci_loc_dam['infrastructure_group'].isna()]], ignore_index=True)
        ## keep only records which are actually about CI (e.g., not theatre, stadion ..)
        df_resp_ci_loc_dam.dropna(subset=["infrastructure_group"], inplace=True)


        # clean up after each chunk
        gc.collect()
        torch.cuda.empty_cache()  # mainly needed after training, small effect when LLM applied only for inference
        torch.no_grad()


        # disentangle rows which contain multiple locations (e.g. "and")
        df_resp_ci_loc_dam["location_orig"] = df_resp_ci_loc_dam["location"] # copy for tracing back 
        # for i in [" and ", " or ", "\bto ", ", ", "; "]:
        #     df_resp_ci_loc_dam = pp.split_text_into_multiple_rows(df_resp_ci_loc_dam, "location",  split_at = i)
    
    
        ## clean-up locs from unneeded text: "the", "between",
        df_resp_ci_loc_dam["location"] = pp.postprocess_locations(df_resp_ci_loc_dam, "location")
        

        print(f"### ---- Processing time for chunk {chunk_no+1} STEP 1: {np.round((time.time() - time2) / 60, 1)} minutes ---- ###")

        # saving step 1 for overall eval
        df_resp_step1_end = pd.concat([df_resp_step1_end, df_resp_ci_loc_dam], ignore_index=True)


        print("\nGEOLLAMA")

        print(f"Starting geoparsing")

        time_geoparse = time.time()


        ## Start geoparsing with geollama to verify Location        
        
        # # extract locations
        # # FIXME             
        # ## replace e.g. and i.e. --> eg and ie to avoid sentence splits
        # chunk.text = re.sub(r"e\.g\.\s+", "eg ", chunk.text)
        # chunk.text = re.sub(r"i\.e\.\s+", "ie ", chunk.text)


        ## TEST if geollama finds more LOCs (eg. rivers) when applied on sentence level
        if use_geollama: 
            
                
            df_geollama_resp = pd.DataFrame()

            resp = geo_llama.geoparse(chunk.text)        
            # df_geollama_resp = pd.DataFrame(resp[0:])
            print(f"Time for geoparsing took {np.round((time.time() - time_geoparse) / 60, 1)}" )
        
            # sentences = []
            # for s in chunk.text.split(". "):
            #     ## aaply geollama on sentences and append response to DF
            #     resp = geo_llama.geoparse(s)
            #     df_geollama_resp = pd.concat([df_geollama_resp, pd.DataFrame(resp[0:])], ignore_index=True)
            #     sentences.append(s)
            # print(f"Time for geoparsing took {np.round((time.time() - time_geoparse) / 60, 1)}" )

            # ## merge single sentences back and add punctuations back for e.g. and i.e.
            # chunk.text = ". ".join(sentences)
            # chunk.text = re.sub(r"eg\s+", "e.g. ", chunk.text)
            # chunk.text = re.sub(r"ie\s+", "i.e. ", chunk.text)


            result_e =  [{
                    "loc_name": obj["name"],
                    "osm_key": obj["properties"]["geocoding"]["osm_key"],
                    "osm_value": obj["properties"]["geocoding"]["osm_value"],
                    "geometry": shape(obj["geometry"])
                } for obj in resp if obj is not None]

            df_geollm = gpd.GeoDataFrame(result_e)

            # add metadata to geollama response per chunk
            df_geollm["citation_id"] = citation
            df_geollm["chunk_id"] = chunk_no
            df_geollm["chunk_text"] = chunk.text

            # maybe empty response
            try:
                # print("Removing duplicated locations and countries from geollama response to keep only more specific location info")
                print("Removing duplicates, and keep countries in the geollama response")

                df_geollm["loc_name"] = list(set(df_geollm["loc_name"]))  # rm dublicates
                print("Toponyms from geollama\n", df_geollm["loc_name"])
            except:
                pass

            df_geollama_resp = pd.concat([df_geollama_resp, df_geollm], ignore_index=True)


            try:
                locations_per_chunk = df_geollm["loc_name"].to_list()
                geometries_per_chunk = df_geollm["geometry"].to_list()
                osm_key_per_chunk = df_geollm["osm_key"].to_list()
                osm_value_per_chunk = df_geollm["osm_value"].to_list()
            except:
                print("No locations extracted by geollama for this chunk. Going to llm-judge\n", df_geollm[0:])
                locations_extracted = False
                # continue

            # sanity check that only cases with location infos are considered for further processing      
            if len(locations_per_chunk) == 0:
                print("geollama did not find any potential locations found for this chunk. Going to llm-judge")
                locations_extracted = False
                # continue

            # only process when Locations/coords where found
            if locations_extracted: 
                print("Enriching LLm 1 response - with potential coords")
                
                coord_potential_locations = dict(
                    zip(locations_per_chunk, 
                        zip(
                            geometries_per_chunk,
                            osm_key_per_chunk,
                            osm_value_per_chunk
                )))
                df_resp_ci_loc_dam["coord_potential_locations"] = [
                    coord_potential_locations
                ] * len(df_resp_ci_loc_dam)

                print("Enriching LLm 1 response - find corresponding coords")

                df_resp_ci_loc_dam["coords"] = None
                df_resp_ci_loc_dam["coords_coarse"] = None

                # reindex to avoid issues in iloc
                df_resp_ci_loc_dam.reset_index(drop=True, inplace=True)


                ## get most likely coordinates for locations
                counter_errors = 0
                for entry in range(len(df_resp_ci_loc_dam)):

                    loc = df_resp_ci_loc_dam["location"].iloc[entry]
                    loc_potential_coords = df_resp_ci_loc_dam["coord_potential_locations"].iloc[entry]
                    try:
                        # go to next entity when it is NAN
                        if isinstance(loc, float):
                            continue
                        if loc == np.nan or loc == "nan" or loc == "NAN" or loc == "NaN" or loc == "":
                            continue
                        
                        # write coordinates to df
                        df_resp_ci_loc_dam.at[entry,  "coords"] = shape(loc_potential_coords[loc][0])

                    except Exception as e:
                        try:
                            for coords in loc_potential_coords.keys(): 

                                ## WORKAROUND handling "Ahr river valley" <-> "Ahrtal"
                                if coords == "Ahrtal":  # geollama response
                                    df_resp_ci_loc_dam.at[entry, "location"] = df_resp_ci_loc_dam.at[entry,"location"].replace("Ahr River valley", "Ahrtal")
                                    loc = df_resp_ci_loc_dam["location"].iloc[entry]

                                smlrty = fuzz.partial_ratio(loc, coords)
                                if smlrty > 90: 
                                    print("\nUsing partial ratio for matching:", loc, "<->", coords, f"{smlrty}")
                                    # NOTE: handles "Malaga airport", "the Ahr valley", "Erft region"
                                    # FIXME needs imrovement in the future, to get more concrete spat. info (if it is region, a river etc.)
                                    print("loc_potential_coords[coords] 2", loc_potential_coords[coords])
                                    df_resp_ci_loc_dam.at[entry,  "coords"] = shape(loc_potential_coords[coords][0])
                                else: pass
                    
                        except Exception as e:
                            print(f"\nEntry {entry}: {df_resp_ci_loc_dam['location'].iloc[entry]}")
                            print(f"No geolocalization possible for row {entry}: {e} \n{df_resp_ci_loc_dam['coord_potential_locations'].iloc[entry]}")
                            # e.g. "flood region", "A76 in both directions"
                            print("Creating BBox of potential location based on locations mentioned in respective chunk text")
                            coords_list = [[float(v[0]), float(v[1])] for v in loc_potential_coords.values()]
                            df_resp_ci_loc_dam.at[entry,  "coords_coarse"] =  get_bbox(coords_list)
                            counter_errors = counter_errors + 1

                # postprocess to geoDF
                df_resp_ci_loc_dam.drop(columns=["coord_potential_locations"], inplace=True)
                df_resp_ci_loc_dam = gpd.GeoDataFrame(df_resp_ci_loc_dam, geometry="coords")

                print(f"{counter_errors} Cases where only coarse loc could be extracted ( based on loc in entire chunk):")
                ## drop cases where no geolocalization could be done 

                df_resp_step2 = pd.concat([df_resp_step2, df_resp_ci_loc_dam],ignore_index=True)           

                print(f"calling LLM 2 took {(time.time() - time_geoparse) / 60} min" )


                    # enrich LLM response for each chunk
                    # df_resp_2["citation_id"] = citation
                    # df_resp_2["chunk_id"] = chunk_no  # add chunk id as identifier
                    # # df_resp_2["ci_entity"] = df_ci_geo_chunk["ci_entity"] 
                    # # df_resp_2["geo_entity"] = df_ci_geo_chunk["geo_entity"]
                    # df_resp_2["coord_potential_locations"] = str(dict(zip(locations_per_chunk, zip(lats_per_chunk, lons_per_chunk, RAGestimated_per_chunk)))) # INTERIM for verification of lat, lon 
                    # df_resp_2["infrastructure_type_org"] = df_resp_ci_loc_dam_step1["infrastructure_type"] 
                    # df_resp_2["damage_org"] = df_resp_ci_loc_dam_step1["damage"] 
                    # df_resp_2["damage_value_org"] = df_resp_ci_loc_dam_step1["damage_value"] 
                    # df_resp_2["location_org"] = df_resp_ci_loc_dam_step1["location"] 
                    # df_resp_2["location_type_org"] = df_resp_ci_loc_dam_step1["location_type"] 
                    # df_resp_2["chunk_text"] =  context[0]["text"]  # add (translated) chunk text for tracing back LLM response

                # # collect resps for each doc
                print("CREATED final LLM response (STEP 1 & 2) successfully")
                # print(df_resp_2)

                # except (IndexError, ValueError, KeyError) as e:
                #     print(f"Cannot add response: {e},\nFaulty response (before postprocessing):", response)
                #     # faulty response: e.g.  .., "location": "V" on satellite and online on radar"}, { ...}, {}
                #     responses_error_list.append({
                #         "citation_id": citation,
                #         "chunk_id": chunk_no,
                #         "response": response, #.replace('\n', ''),
                #         "error": str(e)
                #     })

            print(f"\n   Processing time for chunk {chunk_no+1} STEPs 1 & 2: {np.round((time.time() - time2) / 60, 1)} minutes\n")

    
        # clean up before applying CUDA
        gc.collect()
        torch.cuda.empty_cache() 
        print(torch.cuda.memory_reserved() / 1e9)
        torch.no_grad()


        print("\nLLM-AS-JUDGE")


        # LLM judge - step3
        # sanity check: remove societal/environmental amounts whose numbers do not occur in the chunk text, before the judge sees them
        df_resp_ci_loc_dam = pp.remove_amounts_not_in_text(df_resp_ci_loc_dam, "societal_impact_amount", "chunk_text")
        # df_resp_ci_loc_dam = pp.remove_amounts_not_in_text(df_resp_ci_loc_dam, "environmental_impact_amount", "chunk_text")

        print("\nLLM-as-a-judge: checking plausibility of extracted records based on chunk text")
        time_judge = time.time()
        df_resp_ci_loc_dam = judge_records(df_resp_ci_loc_dam, chunk.text, llm_judge)
        print(lj.format_judge_report(df_resp_ci_loc_dam, verdicts="adapted"))  # verdict counts + adapted records (old -> new)
        df_resp_judged = pd.concat([df_resp_judged, df_resp_ci_loc_dam], ignore_index=True)
        print(f"calling LLM-as-a-judge took {np.round((time.time() - time_judge) / 60, 1)} min")



    print(f"\n   Processing time for document {citation} STEP 1&2: {np.round((time.time() - time1) / 60, 1)} minutes\n")

    print(f"Safety: saving responses (Step 1 + 2) for doc: {citation} ")
    # chain of prompts
    df_resp_step1_end.to_csv(f"../../interim_results_debug/llm_geollm_chain_step1_{citation}.csv", encoding='utf-8', index=False)
    df_resp_step2.to_csv(f"../../interim_results_debug/llm_geollm_chain_step2_{citation}.csv", encoding='utf-8', index=False)
    df_resp_judged.to_csv(f"../../interim_results_debug/llm_geollm_chain_judge_{citation}.csv", encoding='utf-8', index=False)
    
    
    # errors of all documents so far (incl. chunks skipped due to LengthLimitError), overwritten after each document
    df_errors = pd.DataFrame(responses_error_list, columns=["citation_id", "chunk_id", "response", "error"])
    df_errors["response"] = df_errors["response"].astype(str)  # graph results / pydantic objects -> readable text
    df_errors.to_csv(f"../../interim_results_debug/llm_responses_errors_{citation}.csv", encoding='utf-8', index=False)
    print(f"! WARN {len(df_errors)} faulty or skipped chunks so far, of these {df_errors['error'].str.startswith('LengthLimitError').sum()} due to the token limit")

    # # clean up after each document
    # gc.collect()
    # torch.cuda.empty_cache()  # mainly needed after training, small effect when LLM applied only for inference
    # torch.no_grad()

    # # collecting all docs
    # df_resp_step1_all = pd.concat([df_resp_step1_all, df_resp_step1], ignore_index=True)
    # df_resp_step2_all = pd.concat([df_resp_step2_all, df_resp_step2], ignore_index=True)


    # # # Turn profiling off
    # scalene_profiler.stop()
    # scalene_profile_2.stop()


## clean model memory  
print("Cleaning OpenAi model memory")
# del  decoder_model_1
# del  decoder_model_2
del  graph



print(f"\n\n ######## -------- CI impact extraction took {(time.time() - time0) / 60} minutes -------- ######## \n\n")

# with cascading (e, s,second_ci) + single calls of LLm per CI, Loc, DAM = 70min
# without cascading (e, s,second_ci) + one combi-call of CI_LOC_DAM= 57 min
# with cascading (e, s,second_ci) + one combi-call of CI_LOC_DAM= 80 min
# with geollm + cascading (e, s,second_ci) + one combi-call of CI_LOC_DAM + is_ci_case=T/F = 88 min
# no geollm + casca (detialed prompts) +  direct_qunati_no + one combi-call of CI_LOC_DAM + is_ci_case=T/F: 133.0 minutes (local machine)

# if not RUN_EVALUATION:
#     print("Extraction finished, evaluation skipped (RUN_EVALUATION = False)")
#     sys.exit(0)
# %%
df_resp_step1_end.to_csv(f"../../interim_results_debug/llm_geollm_chain_step1incomplete_{citation}.csv", encoding='utf-8', index=False)


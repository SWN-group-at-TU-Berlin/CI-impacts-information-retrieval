# %% [markdown]
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
import subprocess
import re
import time
from glob import glob
from pathlib import Path
import importlib.util
from itertools import chain

import numpy as np
import pandas as pd
import spacy
from jinja2 import Template
import langextract as lx
import textwrap
from langchain_docling import DoclingLoader
from huggingface_hub import login
import torch

from pdfminer.high_level import extract_text
from docling_core.types.doc.document import TextItem
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.document_converter import DocumentConverter, FormatOption

print("loading settings ..")
sys.path.append("./")
from src.settings import settings as s
from src.document_cleaning import remove_references, remove_headers_footers

torch.manual_seed(42)

# set default location to store model before loading transformers
os.environ["HF_HOME"] = (
   "/home/users/a/a-buch/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror"
    # "/home/a-buch/Documents/TUB_DWN/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror/"
)

# default ollama model
MODEL_CHOICE = "llama3" #"llama3.2:1b"  # The smallest llama model


## NOTE: make sure to set project root as working dir
DOCS_DIR = s.PATH_DATA + "text_sources/"
PARSED_TEXT_DIR = s.PATH_DATA + "parsed_documents/"
md_dir = Path(PARSED_TEXT_DIR)
md_dir.mkdir(parents=True, exist_ok=True)

OUTPUT_LX_DIR = s.PATH_DATA + "langextract_output/"
os.makedirs(OUTPUT_LX_DIR, exist_ok=True)


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


# %%
print("Try loading spaCy model from .cache ...")
try: 
    nlp = spacy.load(s.SPACY_MODEL)

except (OSError, ValueError):
    print(f"spaCy language model '{s.SPACY_MODEL}' not found. Downloading ...")

    # except SyntaxError as e:
    print("Try loading spaCy model for remote environment")
    subprocess.check_call(["uv", "pip", "install", "spacy-transformers"])
    subprocess.check_call(["uv", "run", "python", "-m", "spacy", "download", s.SPACY_MODEL])
    nlp = spacy.load(s.SPACY_MODEL)



# %% [markdown]
# # Document cleaning

# %%
# Docling pipeline configs
accelerator_options = AcceleratorOptions(
    num_threads=4, device=AcceleratorDevice.AUTO
)  # use GPU + multi-threading
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = (
    True  # identify tables as such just not to have them in the TextItems later
)
pipeline_options.accelerator_options = accelerator_options
pipeline_options.force_backend_text = True



# %%
# setup converter for PDF and markdown
converted = DocumentConverter(
    allowed_formats=[InputFormat.PDF, InputFormat.MD],
    format_options={
        InputFormat.PDF: FormatOption(
            pipeline_cls=StandardPdfPipeline,
            pipeline_options=pipeline_options,
            backend=PyPdfiumDocumentBackend,
        ),
    },
)

# %%


# %%
print( "Number of documents to process:", len(os.listdir(DOCS_DIR)) )


# convert the different layouts of the pdf files into unified markdown format incl. sub/section titles, tables, caption text etc
for pdf_filename in os.listdir(DOCS_DIR):
    if pdf_filename.endswith(".pdf"):

        md_filename = f"{Path(pdf_filename).stem}.md"

        pdf_filepath = os.path.join(DOCS_DIR, Path(pdf_filename))
        md_filepath = os.path.join(PARSED_TEXT_DIR, Path(md_filename))
        cleaned_md_filepath = md_filepath.replace(".md", "_cleaned.md")

        if os.path.exists(md_filepath):
            print(
                f"Markdown file '{md_filepath}' already exists. Skipping conversion and cleaning."
            )
            continue

        start_time = time.time()
        print(f"\nFetching: {pdf_filename}")

        print("Remove reference section")
        pdf_text = extract_text(pdf_filepath)
        pdf_text_no_references = remove_references(pdf_text)

        # FIXME remove workaround of saving pdf as markdown and reading it again as Docling.Document
        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(pdf_text_no_references)

        # FIXME with DocLoader
        # loader = DoclingLoader(md_filepath)
        # md_text = loader.load()
        print("Converting Markdown to text...")
        md_text = converted.convert(md_filepath)

        print("Removing headers and footers\n")
        md_text_cleaned = remove_headers_footers(md_text)

        print("Removing URLs") # LangExtract tries to open these URLs when they occur in the document text        
        # text_items = [x for x in md_text_cleaned.document.texts if isinstance(x, TextItem)]
        # for i in range(len(text_items)):
        #     re.sub(r"http\S+", "", text_items[1].orig) # TODO make as func

        # md_text_cleaned = re.sub(r"http\S+", "", md_text_cleaned.document.texts) # TODO make as func
        
        # if doc[i].page_content.strip():          # skip paragraph when it is empty
                #     continue

        print(f"Saving parsed and cleaned document as markdown to: {cleaned_md_filepath}")
        md_text_cleaned.document.save_as_markdown(cleaned_md_filepath)

        end_time = time.time() - start_time
        print(f"Parsing and cleaning done. Time elapsed: {end_time:.2f} seconds.")


# visual check of removed items
# TODO make as document_cleaning function: print removed items with largest number of chars first
# ## NOTE. high number of chars == more pontetially actual text body

# text_items_removed = sorted(text_items_to_drop_visualization, key=lambda x: -x[0])
# for i in text_items_removed[:50]:
#     print(i) # -->  also subsection titles were removed partly




# %% [markdown]
# # LLama with LangExtract

# %%


# %% [markdown]
# ##  Test Llama 3 loaded with Ollama and applied in LangExtract
# 
# Note. decided for llama instead of Mistral or GPT-J, as the former is probably quite similar in its performance (and easy to setup in Ollama) and the later is too much outdated compared to Llama 3
# 
# 
# Langextract does not support Meta models (eg llama) directly only via Ollama, thus we need an API key first to start our Ollama server :)
# 
# Steps to do to when we want to run Ollamas llama/mistral model inside a docker container. Benfits are it makes the installation independent of the local machine + later facilitates distributability of our software
# * Setup ollama for docker + GPU setings by installing [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation ) which we need to run Ollama model with my GPU inside a docker container
# * then configure docker to use nvidita driver via using nvidia toolkit - if possible in rootless mode which means your containers and docker daemon can be run without root user privileges (`nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json`). Then restart your docker daemon (for detailed documentation, see [here](https://medium.com/cyberark-engineering/how-to-run-llms-locally-with-ollama-cb00fa55d5de ) and [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuration)): 
# ```
# nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json
# systemctl --user restart docker
# sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
# ```  
# 
# * start the container: `docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama`, if needed with superuser rights. Note. the first ollama refers to the container name, the second to the client name- See also [here](https://hub.docker.com/r/ollama/ollama)
# * check if the ollama server is running on REST API, by checking message for `http://localhost:11434/` in your web browser
# * Then pull the model you want to use , e.g `ollama pull llama3` and the `ollama serve` (see, [here](https://github.com/google/langextract?tab=readme-ov-file#api-key-setup-for-cloud-models)), When you use ollama in a container use: `docker exec -it <containername> ollama pull llama3` , then `docker exec -it <containername> ollama serve`  . In case you get the info that the server port(here 11434) is already in use, check the processes on this port (`sudo lsof -i :11434`) and kill it, maybe you also need to stop ollama server: `systemctl stop ollama` and/or reload the daemon (see similar issue discussed [here](https://github.com/ollama/ollama/issues/707))
# * In case a container with an image already exists: simply run `docker compose up -d`
# 
# * Lastly, implement pulled model in LangExtract framework:
# ```
# # test with local model from Ollama
# result = lx.extract(
#     text_or_documents=input_text,
#     prompt_description=prompt,
#     examples=examples,
#     model_id="llama3",  # Automatically selects Ollama provider
#     model_url="http://localhost:11434",  # where our Ollama server is running 
#     fence_output=False,
#     use_schema_constraints=False
# )
# ```
# 
# Note:
# * In the examples above, the model runs on localhost, port: 11434
# * Optional: pull model and run in terminal, e.g. llama-3, in container named ollama. `docker exec -it ollama ollama run llama3`
# * Get your Ollama key [here](https://signin.ollama.com/?client_id=client_01JX0QMHD43PFFCCNXH82A6K8B&redirect_uri=https%3A%2F%2Follama.com%2Fauth%2Fcallback&authorization_session_id=01KC98BGZP7TJNPGYZE1P27SC6)
# * check which models are available in Ollama. `$ curl https://ollama.com/api/tags` or via [Ollama documentation](https://ollama.com/library/)
# 
# For more info see also: 
# * Batch processing and long texts or about functioning of langExtract: [Weights & Biases ](https://wandb.ai/wandb_fc/genai-research/reports/LangExtract-Transform-text-into-structured-data-with-AI--VmlldzoxNDI1OTMyNw#:~:text=LangExtract%20is%20open%2Dsource%20and,without%20requiring%20any%20fine%2Dtuning)
# 

# %% [markdown]
# ### Prompt and few-shot examples
# 

# %%
# # 1. Define the prompt and extraction rules
# prompt = textwrap.dedent(
#     """
#     Extract information from the context about the affected infrastructure_type, its damage, its geolocation, as well as about cascading impacts to other infrastructure assets.
#     Extract also information about the societal or economic impacts which resulted from the disrupted infrastructure.

#     Use the exact text for extractions. DO NOT paraphrase or overlap entities.
#     Provide meaningful attributes for each entity to add context.

#     Provide in the field "damage" the type of damage to the specific infrastructure.
#     If no information about the damage type is found, then return for this field a "NAN" value.
        
#     Provide in the field "impacts_to_other_infrastructure_assets" information about cascading impacts to other infrastructure assets mentioned in the context.
#     If no information about cascading impacts to other infrastructure assets is found, then return for this field a "NAN" value.
    
#     Provide in the field "societal_impact" information about societal consequences of the infrastructure failures mentioned in the context.
#     If no information about societal consequences is found, then return for this field a "NAN" value.
    
#     Provide in the field "economic_impact" information about economic consequences of the infrastructure failures mentioned in the context.
#     If no information about economic consequences is found, then return for this field a "NAN" value.
    
#     Finally, evaluate and improve your answer.
    
#     """
# )

# # 2. Provide some high-quality examples to guide the model
# few_shot_examples = [
#     lx.data.ExampleData(
#         text="More than 130 km of motorways were closed directly after the event, of which 50 km were still closed two months later, with an estimated repair cost of EUR100 million (Hauser, 2021). ",
#         extractions=[
#             lx.data.Extraction(
#                 extraction_class="infrastructure_type",
#                 extraction_text="motorways",
#                 attributes={"time": "directly after the event"},
#             ),
#             lx.data.Extraction(
#                 extraction_class="economic_impact",
#                 extraction_text="EUR100 million",
#                 attributes={"type": "repair cost"},
#             ),
#         ],
#     ),
#     lx.data.ExampleData(
#         text="Of the 112 bridges in the flooded 40 km of the Ahr valley (Rhineland-Palatinate), 62 bridges were destroyed, 13 were severely damaged and only 35 were in operation a month after the flood event (MDR, 2021).",
#         extractions=[
#             lx.data.Extraction(
#                 extraction_class="infrastructure_type",
#                 extraction_text="bridges",
#                 attributes={"damage type": "destroyed", "number": "62"},
#             ),
#             lx.data.Extraction(
#                 extraction_class="geolocation",
#                 extraction_text="Ahr valley",
#                 attributes={"region": "Rhineland-Palatinate"},
#             ),
#         ],
#     ),
#     lx.data.ExampleData(
#         text=" In total, at least 220 casualties have been reported, with insured loss estimates of approximately EUR 150 million–EUR 250 million in the Netherlands (Verbond voor Verzekeraars, 2022), "
#         "EUR 2.2 billion in Belgium (Assuralia, 2022) and EUR 8.2 billion (GDV, 2022) in Germany. "
#         "The event caused major damages to residential and commercial structures and to many critical infrastructure (CI) assets. ",
#         extractions=[
#             lx.data.Extraction(
#                 extraction_class="economic_impact",
#                 extraction_text="EUR 150 million–EUR 250 million",
#                 attributes={
#                     "type": "insured loss estimates",
#                     "geolocation": "Netherlands",
#                     "citation": "Verbond voor Verzekeraars, 2022",
#                 },
#             ),
#             lx.data.Extraction(
#                 extraction_class="economic_impact",
#                 extraction_text="EUR 2.2 billion",
#                 attributes={
#                     "type": "insured loss estimates",
#                     "geolocation": "Belgium",
#                     "citation": "Assuralia, 2022",
#                 },
#             ),
#             lx.data.Extraction(
#                 extraction_class="economic_impact",
#                 extraction_text="EUR 8.2 billion",
#                 attributes={
#                     "type": "insured loss estimates",
#                     "geolocation": "Germany",
#                     "citation": "GDV, 2022",
#                 },
#             ),
#         ],
#     ),
# ]

# %%


# %%
# 1. Define the prompt and extraction rules
prompt = textwrap.dedent(
    """
    Extract information from the context about the affected infrastructure_type, its damage, its geolocation, as well as about cascading impacts to other infrastructure assets.
    Extract also information about the societal or economic impacts which resulted from the disrupted infrastructure.

    Use the exact text for extractions. DO NOT paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.

    Provide in the field "damage" the type of damage to the specific infrastructure.
    If no information about the damage type is found, then return for this field a "NAN" value.
        
    Provide in the field "impacts_to_other_infrastructure_assets" information about cascading impacts to other infrastructure assets mentioned in the context.
    If no information about cascading impacts to other infrastructure assets is found, then return for this field a "NAN" value.
    
    Provide in the field "societal_impact" information about societal consequences of the infrastructure failures mentioned in the context.
    If no information about societal consequences is found, then return for this field a "NAN" value.
    
    Provide in the field "economic_impact" information about economic consequences of the infrastructure failures mentioned in the context.
    If no information about economic consequences is found, then return for this field a "NAN" value.
    
    Finally, evaluate and improve your answer.
    
    """
)

# 2. Provide some high-quality examples to guide the model
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
            lx.data.Extraction(
                extraction_class="impacts_to_other_infrastructure_assets",
                extraction_text="transportation",
                attributes={"damage": "disrupted", "geolocation": "Milan (city)"},
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
                attributes={"damage": "affected", "geolocation": "Partinico area", "name": "State highway 113"},
            ),
        ],
    ),
    # lx.data.ExampleData(
    #     text="Further damage occurred in recent days at Catania airport, considered the fifth most important in Italy."
    #         "Following a fire that broke out inside the terminals, the access areas were promptly closed to travelers, causing enormous damage to the local economy, the tourism sector, and various professionals."
    #         "In monetary terms, the damage is enormous: the National Civil Aviation Authority estimates a total investment for facilities and maintenance of around €200,000."
    #         "The MEC (Consumer Voters Movement), on the other hand, estimates a cost of around €40 million per day due to the closure of the airport as a result of the fire." 
    #         "The total damage is estimated at more than €80 million. The fire broke out last Sunday night at Vincenzo Bellini Airport in Catania and, after two days of relentless work to extinguish the fire, arrivals and departures resumed from Terminal C.",
    #     extractions=[
    #         lx.data.Extraction(
    #             extraction_class="infrastructure_type",
    #             extraction_text="terminals",
    #             attributes={
    #                 "damage": "damaged",
    #                 "geolocation": "Catania",
    #                 "name": "Vincenzo Bellini Airport"
    #             },
    #         ),
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
    #     ],
    # ),

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
                extraction_text="reopening", #"Port operability",
                attributes={"damage": "temporarily closed", "geolocation": "Valencia and Sagunto ports"},
            ),
            # lx.data.Extraction(
            #     extraction_class="impacts_to_other_infrastructure_assets",
            #     extraction_text="reopening", #" Port operability",
            #     attributes={"damage": "temporarily closed", "geolocation": "Sagunto port"},
            # )
        ],
    ),
    # ## Khazai et al 2013
    # lx.data.ExampleData(
    #     text="Durch einen Deichbruch am 10.06. mussten im Landkreis Stendal Fernverkehrsstrecken der Deutschen Bahn AG gesperrt werden, sodass es zu Zugausfällen und langen Verspätungen kommt.",
    #     extractions=[   # tricky extraction due to non-English language
    #         lx.data.Extraction(
    #             extraction_class="infrastructure_type",
    #             extraction_text="Deichbruch",
    #             attributes={"damage": "dam failure", "geolocation": "Stendal (Landkreis)"},
    #         ),
    #         lx.data.Extraction(
    #             extraction_class="impacts_to_other_infrastructure_assets",
    #             extraction_text="Fernverkehrsstrecken",  # "high-speed network",
    #             attributes={"damage": "closure"},
    #         ),
    #         # lx.data.Extraction(
    #         #     extraction_class="impacts_to_other_infrastructure_assets",
    #         #     extraction_text="Verspätungen",   # train service
    #         #     attributes={"damage": "disrupted"},
    #         # ),
    #         lx.data.Extraction(
    #             extraction_class="impacts_to_other_infrastructure_assets",
    #             extraction_text="Zugausfällen",  # train service
    #             attributes={"damage": "cancellations"},
    #         )
    #     ],
    # ),
    # ## Koks et al., 2022
    # lx.data.ExampleData(
    #     text="In Belgium, several towns experienced disruptions in water supply (in particular as a result of pollution).",
    #     extractions=[
    #         # lx.data.Extraction(
    #         #     extraction_class="infrastructure_type",
    #         #     extraction_text="water", # "water network",
    #         #     attributes={"damage": "contaminated", "geolocation": "Belgium"},
    #         # ),
    #         lx.data.Extraction(
    #             extraction_class="impacts_to_other_infrastructure_assets",
    #             extraction_text="water supply",
    #             attributes={"damage": "disrupted"},
    #         ),
    #     ],
    # ),

    # # 1. LAST ONE CHANGED:
    # lx.data.ExampleData(
    #     text="In Belgium, several towns experienced disruptions in water supply (in particular as a result of pollution)." 
    #     "Directly after the event, approximately 3400 families had no access to potable water.",
    #     extractions=[
    #         lx.data.Extraction(
    #             extraction_class="infrastructure_type",
    #             extraction_text="water supply",
    #             attributes={"damage": "polluted", "geolocation": "Belgium"},
    #         ),
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
    #     ],
    # ),
    lx.data.ExampleData(
        text="Within the region of Rhineland-Palatinate, it took 2 weeks to ensure 100 % coverage again through emergency communication masts." 
        "Within 1 month, most of the network was restored to pre-disaster service provision." 
        "After 5 months, broadband has also been restored in the most affected areas, which started in most areas only after power infrastructure was rebuilt.",
        extractions=[
            lx.data.Extraction(
                extraction_class="infrastructure_type",
                extraction_text="power infrastructure",
                attributes={"damage": "affected", "geolocation": "Rhineland-Palatinate"},  # tricky extraction of location-info
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
            lx.data.Extraction(
                extraction_class="economic_impact",
                extraction_text="EUR100 million",
                attributes={"damage": "repair cost"},
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
                    "geolocation": "Ahr valley", "region": "Rhineland-Palatinate"
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text=" In total, at least 220 casualties have been reported, with insured loss estimates of approximately EUR 150 million–EUR 250 million in the Netherlands (Verbond voor Verzekeraars, 2022), "
        "EUR 2.2 billion in Belgium (Assuralia, 2022) and EUR 8.2 billion (GDV, 2022) in Germany. "
        "The event caused major damages to residential and commercial structures and to many critical infrastructure (CI) assets. ",
        extractions=[
            lx.data.Extraction(
                extraction_class="economic_impact",
                extraction_text="EUR 150 million–EUR 250 million",
                attributes={
                    "damage": "insured loss estimates",
                    "geolocation": "Netherlands",
                    "citation": "Verbond voor Verzekeraars, 2022",
                },
            ),
            lx.data.Extraction(
                extraction_class="economic_impact",
                extraction_text="EUR 2.2 billion",
                attributes={
                    "damage": "insured loss estimates",
                    "geolocation": "Belgium",
                    "citation": "Assuralia, 2022",
                },
            ),
            lx.data.Extraction(
                extraction_class="economic_impact",
                extraction_text="EUR 8.2 billion",
                attributes={
                    "damage": "insured loss estimates",
                    "geolocation": "Germany",
                    "citation": "GDV, 2022",
                },
            ),
        ],
    ),
]

# %% [markdown]
# ### Appy LangExtract

# %%
# doc[i].page_content
# context_example = "In mid-July 2021, a persistent low-pressure system caused extreme precipitation in parts of the Belgian, German and Dutch catchments of the Meuse and Rhine rivers. This led to record-breaking water levels and severe ﬂooding (Mohr et al., 2022). Comparable heavy precipitation events in this area have never been registered in most of the affected areas before (Kreienkamp et al., 2021). The German states most affected include Rhineland-Palatinate (Rheinland-Pfalz), with damage to the Ahr River valley (Ahrtal), several regions in\n\nthe Eiffel National Park, to the city of Trier. Flooding in Belgium was concentrated in the Vesdre River valley (districts of Pepinster, Ensival and Verviers), the Meuse River valley (Maaseik, Liége), the Gete River valley (Herk-de-Stad and Halen) and southeast Brussels (Wavre). The Netherlands experienced ﬂooding, mostly concentrated in the southern district of Limburg. In total, at least 220 casualties have been reported, with insured loss estimates of approximately EUR 150 million–EUR 250 million in the Netherlands (Verbond voor Verzekeraars, 2022), ∼ EUR 2.2 billion in Belgium (Assuralia, 2022) and ∼ EUR 8.2 billion (GDV, 2022) in Germany. The event caused major damages to residential and commercial structures and to many critical infrastructure (CI) assets. Not only vital functions for ﬁrst responders were affected (e.g. hospitals, ﬁre departments), but also railways, bridges and utility networks (e.g. water and electricity supply) were severely damaged, expecting to take months to years to fully rebuild.\n\nCI is often considered to be the backbone of a well-functioning society (Hall et al., 2016), which is particularly eminent during natural hazards and disasters. For instance, failure of electricity or telecommunication services immediately causes disruptions in the day-to-day functioning of people and businesses, including those outside the directly affected area. Despite the (academic) agreement that failure of infrastructure systems may cause (large-scale) societal disruptions (Garschagen and Sandholz, 2018; Hallegatte et al., 2019; Fekete and Sandholz, 2021), empirical evidence on the impacts of extreme weather events on these systems is still\n\nPublished by Copernicus Publications on behalf of the European Geosciences Union.\n\nE. E. Koks et al.: Flood impacts to infrastructure\n\nlimited. This brief communication provides an overview of the observed ﬂood impacts to large-scale infrastructure systems during the 2021 mid-July western European ﬂood event and how reconstruction of these large-scale systems has progressed. Next, we highlight how some of these observations compare to academic modelling approaches. We conclude with suggestions on moving forward in CI risk modelling, based on the lessons learned from this extreme event.\n\nIn Germany, road and railway infrastructure was severely damaged as documented exemplarily in Fig. 1. Cost estimates reach up to EURO 2 billion Euro (MDR, 2021). More than 130 km of motorways were closed directly after the event, of which 50 km were still closed two months later, with an estimated repair cost of EUR 100 million (Hauser, 2021). Of the 112 bridges in the ﬂooded 40 km of the Ahr valley (Rhineland-Palatinate), 62 bridges were destroyed, 13 were severely damaged and only 35 were in operation a month after the ﬂood event (MDR, 2021). Over 74 km of roads, paths and bridges in the Ahr valley have been (critically) damaged. In some cases, repairs are expected to take months to years (Zeit Online, 2021). For example, major freeway sections, including parts of the A1 motorway, were closed until early 2022 (24Rhein, 2022). In addition, about 50 000 cars were damaged, causing insurance claims of some EUR 450 million (ADAC, 2021). The German railway provider Deutsche Bahn expects asset damages of around EUR 1.3 billion. Among other things, 180 level crossings, almost 40 signal boxes, over 1000 catenary and signal masts, and 600 km of tracks were destroyed, as well as energy supply systems, elevators and lighting systems (MDR, 2021). As of 11 April 2022, 14 of the affected rail stretches are fully functional again. The less damaged stretches were functional again within 3 months, while some of the most damaged sections in the Ahr valley are expected to be ﬁnished by the end of 2025 (DB, 2022). In Belgium, approximately 10 km of railway tracks and 3000 sleeper tracks have to be replaced; 50 km of catenary needs to be repaired; and 70 000 t of railway track bed needs to be placed, with estimated costs between EUR 30 million–EUR 50 million (Rozendaal, 2021a). Most damages have been repaired within 2 weeks. The most severely damaged railway line (between the villages of Spa and Pepinster) was reopened again on 3 October 2021 (Rozendaal, 2021b). In the Netherlands, no large-scale damage has been reported to transport infrastructure. A few national highways were partly ﬂooded (e.g. the A76 in both directions) or brieﬂy closed (&lt; 3 d) because of the potential of ﬂooding. Most likely due to relative low-ﬂow velocities, damage to Dutch national road infrastructure was limited. Several railway sections were closed (e.g. the rail-\n\nway section between Maastricht and Liége) and some damage occurred to the railway infrastructure, in particular to the electronic “track circuit” devices and saturated railway embankments (Prorail, 2021).\n\nAt the peak of the event, around 200 000 people experienced power outages in Germany. Electricity infrastructure was severely damaged in North Rhine-Westphalia and Rhineland-Palatinate. However, within 2 d around 50 % of the power was restored through repairs and temporary ﬁxes. Within 8 weeks, no emergency power generators were required anymore, with most of the power infrastructure restored in Germany’s affected areas. Some areas, however, only had permanent power infrastructure after 6 months (Westnetz, 2022). The gas distribution network in the Ahr valley was severely damaged. Approximately 133 km of natural gas pipelines, 8500 gas metres, 3400 house pressure regulators, 7220 of the approximately 8000 household connections, and 31 systems measuring and regulating gas pressure have been damaged or destroyed (SWR, 2021). Gas supply was almost fully restored within 4.5 months after the ﬂood event (Energienetze Mittelrhein, 2021). In Belgium, approximately 41 500 people experienced power outages at the peak of the event. This was the result of both damaged and deliberately switched-off electrical cabinets to prevent serious damages. It took around 3 weeks to fully restore power. Similar to Germany, severe damage had been observed to the gas network. In the villages around Liége, such as Chaudfontaine and Pepinster (Belgium), gas supply was fully recovered within 5 months (Grosjean, 2021; De Wolf, 2021). In the Netherlands, 1000– 2000 households experienced a loss of electricity supply at the peak of the event. Between 100 to 200 households had no gas supply. Within several days, electricity supply was restored (Task Force Fact Finding Hoogwater, 2021).\n\nIn the region of Rhineland-Palatinate (Germany), most drinking water supply was restored within 2 months (Hochwasser Ahr, 2021a). However, sewage treatment plants in Altenahr, Mayschoss and Sinzig had been largely destroyed (Hochwasser Ahr, 2021b), and it is expected to take at least 1.5 years to fully repair most sewage treatment plants. Emergency sewage treatment plants have been built in the meantime (GA, 2021). In the Erft region 7 out of 31 wastewater facilities had been destroyed. Many facilities reported pollution of oil and diesel, forming layers up to 15 cm thick (Kuhn, 2021). In addition, much of the groundwater (and soil) in the ﬂood region was mixed with oil (from destroyed residential oil tanks), chemicals such as fertilizers (from wineries and other agriculture) and chemicals from nearby industrial plants. In Sinzig, 3.6 × 106 L of oil–water mixture was recycled, gaining 3600 m3 of oil, to be reused for heating and\n\nNat. Hazards Earth Syst. Sci., 22, 3831–3838, 2022\n\nE. E. Koks et al.: Flood impacts to infrastructure\n\nFigure 1. Damage in the Ahr valley, Germany (images taken on 11 August 2021). (a) Destruction of federal highway B266 (A1) and railway (A2) near Heimersheim. (b) Further upstream in the Ahr valley (Altenburg), large stretches of the Ahrtalbahn railway have been destroyed (B1) and the few remaining road and rail bridges show signs of temporary repairs (B2). (c) Riverbed erosion uncovered and destroyed many cables supposed to lie more than 80 cm below surface level (C1) as well as sewers (C2). (d) Inundated electricity distribution infrastructure (D1), road erosion and stabilization (D2), uncovered cables (D3), and collapsed buildings in Schuld. Pictures by Margreet van Marle/Deltares/GEER-association, distributed under Creative Commons Attribution 4.0 license.\n\nindustrial usage (Kuhn, 2021). In the heavily destroyed town of Bad Münstereifel (in the state of North Rhine-Westphalia), drinking water supply was re-established within 5 d after the ﬂood event (most frequently through emergency tanks), and about 50 % of the city centre was reconnected to the freshwater network shortly thereafter however, water had to be boiled before consumption until about 1 month later (Bad Münstereifel, 2021). In Belgium, several towns experienced disruptions in water supply (in particular as a result of pollution). Directly after the event, approximately 3400 families had no access to potable water. Within less than a week, this was reduced to around 1650 families (Terzake, 2021). It took, however, 6 months to rebuild the permanent water supply infrastructure (SWDE, 2022). In the Netherlands, little to no problems have been recorded with regards to water supply.\n\nWe found no information regarding direct impact on solid-waste facilities as a result of the ﬂood event. However, there is a large pressure on the solid-waste sector to clean the affected areas; 1 month after the event, we observed dozens of large temporary waste ﬁlls and frequent incidences of oil pollution in Rhineland-Palatinate during a ﬁeld visit. In the Ahrweiler district alone, the ﬂood caused as much solid waste as normally would be collected over 30 years. In Belgium, the amount of solid waste is estimated around 160 000 t, stored at several places, such as the abandoned highway track A601. This highway has been used for approximately 9 months as a temporary storage for debris (Couplez, 2022). In the Netherlands, there have been primarily problems with waste deposits along the river banks, which is mostly the solid waste transported by the river from further upstream. Thousands of tonnes of tree debris (logs and\n\nNat. Hazards Earth Syst. Sci., 22, 3831–3838, 2022\n\nE. E. Koks et al.: Flood impacts to infrastructure\n\nof running water and electricity (Ärzte Zeitung, 2021). After 1.5 months, medical care was guaranteed again in the most affected regions in Rhineland-Palatinate (Hochwasser Ahr, 2021c). In the state of North Rhine-Westphalia, approximately 68 hospitals have been affected, of which several have been affected severely and will take at least 1.5 years to be rebuilt (Fig. 2). Direct damages are estimated to be at least EUR 100 million to repair all medical facilities (Korzilius, 2021). In the town of Eschweiler (Germany), for example, the basement of the hospital was ﬂooded, as well as the outbuildings and the entire outdoor area. The power supply collapsed, the entire building technology was destroyed and some 300 patients had to be evacuated by helicopter. Property damage is expected to be around EUR 50 million. Within 3.5 weeks, the hospital was partly operational, and within 3 months, all hospital operations continued normally (SAH Eschweiler, 2021). The Mutterhaus Ehrang hospital in Trier (Germany) is now permanently closed as the hospital is too severely damaged to rebuild. Furthermore, in the region of Rhineland-Palatinate (Germany), 19 daycare centres and 17 schools suffered damage from the ﬂoods, affecting more than 8000 students (Staib, 2021). Approximately 4 months after the ﬂood event, the district of Bad Neuenahr-Ahrweiler established emergency educational facilities using 297 containers that serve as classrooms, ofﬁces and dining facilities for more than 800 students (Wiesbadener Kurier, 2021). In Belgium, various rural clinics have been affected and were unable to provide any services. Concurrently, in the most affected areas, general-practitioner facilities have been completely destroyed (Le Spécialiste, 2021). In the Netherlands, one nursing home was ﬂooded, and one hospital was evacuated as a precautionary measure.\n\nMost often, large-scale object-based infrastructure impact studies (e.g. Bubeck et al., 2019) only disclose aggregated risk metrics (i.e. country-level risk estimates), which hampers veriﬁcation and validation with observed impacts on smaller scales."


# %%
#


filename= f"{model_name.replace(':', '_')}_all_documents.jsonl"
OUTPUT_LX_FILEPATH = Path(OUTPUT_LX_DIR, filename)


# filename = "Koks et al 2022 Brief communication_cleaned.md"
# FILE_PATH = Path(PARSED_TEXT_DIR, filename)
# filename_stem = FILE_PATH.stem


## extract CI failure impacts via LangExtract

## load docs
docs_list = glob(PARSED_TEXT_DIR + "*_cleaned.md")
print(f"Found {len(docs_list)} cleaned documents ready for LangExtract.")


responses_all_docs = []
responses_doc_ids = [] 

## iterate over all cleaned documents and extract CI-GEO entity pairs
for i, FILE_PATH in enumerate(docs_list):

    filename_stem = Path(FILE_PATH).stem

    print(
        f"\n\n -------- Processing document [{i}]: {Path(FILE_PATH).name} -------- \n"
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

    for i, _ in enumerate(doc):

        # remove URLs
        doc[i].page_content = re.sub(r"http\S+", "", doc[i].page_content) # TODO move to document preprocessing function
        # if doc[i].page_content.strip():          # skip paragraph when it is empty
        #     continue

        try:
            response = lx.extract(
                text_or_documents=doc[i].page_content,
                prompt_description=prompt,
                examples=few_shot_examples,
                model_id= model_name,  # Automatically selects Ollama provider
                
                model_url=os.getenv("OLLAMA_HOST", f"localhost:{host_port}"),  # make explicit where Ollama server is running 
                # model_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),  # make explicit where Ollama server is running 
                # language_model_type=lx.inference.OllamaLanguageModel,
                temperature=0.2,
                extraction_passes=1,  # decrease recall -> faster processing
                # max_workers=4,   # 4 parallel threads
                max_char_buffer=1024,      # adapt to max. token sequence length of model
            )
            responses.append(response)
        except ValueError as e:
            print(f"Error when applying LangExtract on document {filename_stem}: {e}")
            print("Probably one of the few-shot examples does not match")
            pass
        
    # identifiers (i.e. document titles)
    responses_doc_ids.append(filename_stem)

    responses_all_docs.append(responses)


# %%


## save LangExtract output of CI failure impacts
responses_all_docs_unnested = list(chain(*responses_all_docs))

# check if file exists already
if os.path.exists(OUTPUT_LX_FILEPATH):
    print(f"File {OUTPUT_LX_FILEPATH.name} already exists. Not saving to disk.")
    pass
else:
    print(f"Saving LangExtract output to {OUTPUT_LX_FILEPATH}")
    lx.io.save_annotated_documents(
        responses_all_docs_unnested,
        output_name=filename,
        output_dir=OUTPUT_LX_FILEPATH.parent,
    )


# %%


# %%
# safety_copy = responses_all_docs

responses_all_docs.__len__()


# %%
# Display the response from the previous cell's extraction
# response.tokenized_text?
response.tokenized_text.tokens.__len__()


# %% [markdown]
# #### Analyse LangExtract output

# %%
# Look at all extracted entities
for  i in range(len(responses_all_docs_unnested)):
     for j in responses_all_docs_unnested[i].extractions:
          print(f"Type: {j.extraction_class}")
          print(f"Text: '{j.extraction_text}'")
          try:
               print(f"Location: chars {j.char_interval.start_pos}-{j.char_interval.end_pos}")
          except AttributeError:
               print("Location: N/A")
          print(f"Attributes: {j.attributes}")
          print(f"Char span: {j.char_interval}, Token span: {j.token_interval}")
          try:
               print(f"Alignment: {j.alignment_status.value}")
          except AttributeError:
               print("Alignment: N/A")
          
          # write to dataframe
          print("---")

# %%

# # lx.io.save_annotated_documents([result], output_name="../../data/llm_outputs/extraction_results.jsonl", output_dir=".")

# # Generate the visualization from the file
# html_content = lx.visualize(OUTPUT_LX_DIR + f"{model_name.replace(':', '_')}_{filename_stem}.jsonl")
# with open(OUTPUT_LX_DIR + "visualization.html", "w") as f:
#     if hasattr(html_content, 'data'):
#         f.write(html_content.data)  # For Jupyter/Colab
#     else:
#         f.write(html_content)

# %%
responses_all_docs_unnested

# %% [markdown]
# ## Evaluation

# %%


# %%




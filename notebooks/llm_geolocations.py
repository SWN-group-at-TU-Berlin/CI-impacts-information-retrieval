# %% [markdown]
# ## Aim:
# How to link the rule-based extracted CI_TYPE-GEO pairs with the respective Ci failure impacts
# 
# **Idea**\
# Test using prompt engineering by passing table of CI-GEO pairs to GPT-J model.
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
import os

# # settings for CUDA and PYTORCH
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# print(os.environ["CUDA_VISIBLE_DEVICES"])
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["PYTORCH_ALLOC_CONF"]="expandable_segments:True" ## improve memory allocation

# # settings for debugging CUDA errors (pinpoint exact line of error)
os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 

# activate global venv explicitly
os.environ["VIRTUAL_ENV"] = "/beegfs/home/users/a/a-buch/_PROJECTS/CI-impacts-information-retrieval/.venv"


# %%
import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())  # should give 2
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_properties(0))
print(torch.cuda.get_device_properties(1))
print(torch.cuda.get_device_capability())
print(torch.cuda.get_arch_list())
print(torch.__version__)
print(torch.version.cuda)


## --> must be CUDA 12.6, torch: 2.91, ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']


# %%
import os
import sys
import subprocess
import re
import time
from glob import glob
from pathlib import Path
import gc
from io import StringIO
import json

# from tqdm import tqdm
import numpy as np
import pandas as pd
import geonamescache
import pyarrow as pa
import pyarrow.parquet as pq
import langdetect
from langchain_docling import DoclingLoader
import spacy
from huggingface_hub import login
from pdfminer.high_level import extract_text
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.document_converter import DocumentConverter, FormatOption


try:
    print("Trying to import from src and submodules...")
    sys.path.append("./")
    from src.settings import settings as s
    import src.document_cleaning as dc
    import src.translation_model as tm
    import src.extraction_model as em
    import src.postprocess as pp
    import src.utils as u

    from submodules.geollama.src.main import GeoLlama
    from submodules.geollama.src.model import TopoModel, RAGModel
except:
    try:
        print("Cannot import from src or submodules. Check if you are running the notebook from the project root and if the paths to src and submodules are correct.")
        sys.path.append("/home/a-buch/Documents/TUB_DWN/_CLUSTERFOLDER/_PROJECTS/CI-impacts-information-retrieval/src")
        from src.settings import settings as s
        import src.document_cleaning as dc
        import src.translation_model as tm
        import src.extraction_model as em
        import src.postprocess as pp
        import src.utils as u

        from submodules.geollama.src.main import GeoLlama
        from submodules.geollama.src.model import TopoModel, RAGModel
    except:
        print("Still cannot import from src or submodules.")
        sys.path.append("/beegfs/home/users/a/a-buch/_PROJECTS/CI-impacts-information-retrieval/src")
        from src.settings import settings as s
        import src.document_cleaning as dc
        import src.translation_model as tm
        import src.extraction_model as em
        import src.postprocess as pp
        import src.utils as u

        from submodules.geollama.src.main import GeoLlama
        from submodules.geollama.src.model import TopoModel, RAGModel

import transformers
import torch


test_mode = True

# login to HF
login(token=os.getenv("HUGGINGFACE_TOKEN")) 
# NOTE raises exception if not env.variable doesnt exist (compared to os.envrion.get and its shortcut os.getenv)


# NOTE. disabled batch size as OOM for CUDA despite chunkwise memory cleaning, nvtop to find best batchsize
BATCH_SIZE = s.BATCH_SIZE  # max for nvidia GPU

torch.manual_seed(42)

#  automatic linebreaks and multi-line cells.
pd.set_option('display.max_colwidth', 100000)
pd.set_option("display.colheader_justify", "left")

# print(os.environ["CUDA_VISIBLE_DEVICES"])

# clean up before applying CUDA
gc.collect()
torch.cuda.empty_cache() 
print(torch.cuda.memory_reserved() / 1e9)
torch.no_grad()

# check if cuda cna be used
# device = transformers.infer_device()
# print(f"Using device: {device}")


# ## TODO test to prevent CUDA-OOM when reused
# ## Source: https://spacy.io/usage/embeddings-transformers
# from thinc.api import set_gpu_allocator, require_gpu

# # Use the GPU, with memory allocations directed via PyTorch.
# # This prevents out-of-memory errors that would otherwise occur from competing
# # memory pools.
# set_gpu_allocator("pytorch")
## require_gpu(0)



# %%


# %% [markdown]
# ## Set paths and vars

# %%
# set wd to project root
# os.chdir("/home/a-buch/Documents/TUB_DWN/_PROJECTS/CI-impacts-information-retrieval")

## set path variables
DOCS_DIR = Path(s.PATH_DATA / "text_sources/")
PARSED_TEXT_DIR = Path(s.PATH_DATA / "parsed_documents/")
NER_PATTERNS_FILEPATH = Path(s.NER_PATTERNS_FILEPATH)
LLM_OUTPUTS_DIR = Path(s.PATH_DATA / "llm_outputs/")
GEOLLM_OUTPUTS_DIR = Path(s.PATH_DATA / "geollm_outputs/")

os.makedirs(PARSED_TEXT_DIR, exist_ok=True)
os.makedirs(s.PATH_LLM_DATA, exist_ok=True)
os.makedirs(GEOLLM_OUTPUTS_DIR, exist_ok=True)


# CI GEO pairs
CI_GEO_FILEPATH = Path( s.PATH_DATA / s.CI_GEO_PAIRS_FILENAME)

## store LLM 1 response and prompt
OUTPUT_LLM1_FILEPATH =  Path(s.PATH_LLM_DATA / s.LLM_DATA_FILENAME)
OUTPUT_GEOLLM_FILEPATH =  Path(s.PATH_LLM_DATA / "geollm_results.csv")




# %% [markdown]
# ### Set test mode

# %%

docs_list_sample = [
        # Path(PARSED_TEXT_DIR, "AEMET 2024 - ESTUDIO SOBRE LA SITUACIÓN DE LLUVIAS INTENSAS_cleaned.md"),
        # Path(PARSED_TEXT_DIR, "Karakatsani 2023 - Greece economy briefing The economic impact of the recent devastating floods in Greece_cleaned.md"),
        # Path(PARSED_TEXT_DIR, "Lloyd's List 2024 - Port of Valencia reopens after devastating floods_cleaned.md"),
        # Path(PARSED_TEXT_DIR, "Containerlift 2024 - Valencia Port Resumes Operations Following Devastating Flooding in Spain - Containerlift.co.uk - Transport_Lifting_Shipping_cleaned.md"), 
        #     Path(PARSED_TEXT_DIR, "ABC 2024 - Traffic jams and flight delays due to heavy rain and lightning storm in Malaga_cleaned.md"),

        Path(PARSED_TEXT_DIR, "Koks 2022 - Brief communication_cleaned.md"),
        # Path(PARSED_TEXT_DIR, "European Investment Bank 2025 - Spain_ EIB lends €50 million to Iberdrola to rebuild and climate-proof flood-hit power infrastructure in Valencia_cleaned.md"),
        # Path(PARSED_TEXT_DIR, "Wilson 2024 - Flash floods in Spain sweep away cars, disrupt trains and leave several missing _ AP News_cleaned.md"),     
        #     Path(PARSED_TEXT_DIR, "Wildhagen 2013 - Hochwasser_ Wie die Flut Unternehmen lahmlegt_cleaned.md"),

        # # # not Deidda et al, IPCC, Fekete 2025 as it already contains coarse info about many CI impacts
        #     # Path(PARSED_TEXT_DIR, "AFP 2022 - The_Vibes_Valencia Airport in Madrid briefly shut as lightning hits runway _ World _ The Vibes_cleaned.md"),
        #     # Path(PARSED_TEXT_DIR, "Artemis 2015 - PERILS finalises Storm Desmond UK flood loss estimate at £604m_cleaned.md"),
        #     # Path(PARSED_TEXT_DIR, "Brown 2010 - Economy feels chill as UK grinds to a halt _ The Independent_cleaned.md"),
        #     # Path(PARSED_TEXT_DIR, "Diakakis 2020 - A systematic assessment of the effects of extreme flash floods on transportation infrastructure and circulation: The example of the 2017 Mandra flood_cleaned.md"),
        # Path(PARSED_TEXT_DIR, "EFE 2024 - The DANA storm, live_ The death toll rises to 158_cleaned.md"),
        #     # Path(PARSED_TEXT_DIR, "Eurelectric 2006 - Impacts of Severe Storms on Electric Grids_cleaned.md"),
        #     # Path(PARSED_TEXT_DIR, "Euronews 2024 - Spain floods_ Death toll rises to 205 as nation braces for more rain _cleaned.md"),
        # Path(PARSED_TEXT_DIR, "Ferlita 2023 - Incendi in Sicilia, ecco cosa accade_cleaned.md"),
        #     # Path(PARSED_TEXT_DIR, "Fink 2004 - The 2003 European summer heatwaves and drought - synoptic diagnosis and impacts_cleaned.md"),
        # Path(PARSED_TEXT_DIR, "Gilbody Dickerson 2024 - Spain floods_ At least 95 people killed including British man near Malaga _ World News _ Sky News_cleaned.md"),
        #     # Path(PARSED_TEXT_DIR, "Kadir 2014 - The Impact of Natural Disasters on Critical Infrastructures - A Domino Effect-based Study_cleaned.md"),
        #     # Path(PARSED_TEXT_DIR, "Kaur 2025 - Authorities suspect arson in 17 wildfires across Dalmatian coast, Croatia - The Watchers_cleaned.md"),
        #     # Path(PARSED_TEXT_DIR, "Keller 2014 - Mapping Natural Hazard Impacts on Road Infrastructure—The Extreme Precipitation in Baden-Württemberg_cleaned.md"),
        #     # Path(PARSED_TEXT_DIR, "Kettle 2020 - Storm Xaver over Europe in December 2013 Overview of energy impacts and North Sea events_cleaned.md"),
        #     # Path(PARSED_TEXT_DIR, "Koks 2019 - Understanding Business Disruption and Economic Losses Due to Electricity Failures and Flooding_cleaned.md"),
        #     # Path(PARSED_TEXT_DIR, "Korzilius 2021 Nach der Flut_cleaned.md"),

        # # Path(PARSED_TEXT_DIR, "Khazai 2013 - Juni-Hochwasser 2013 in Mitteleuropa - Fokus Deutschland Bericht 2 Auswirkungen und Bewältigung_cleaned.md"),
            
        #     # not part of valid set:
        #     # Path(PARSED_TEXT_DIR, "Krausmann 2014 - STREST report on lessons learned from recent catastrophic events_cleaned.md"), # > 1800 entries LLMv3.0 incl. hallucinations
]


## Test mode
if test_mode:
    search_path = docs_list_sample
    print("Test mode is ON. Using only a small sample of documents for testing.")
else:
    search_path = glob(str(Path(PARSED_TEXT_DIR, "*cleaned.md")))

print(f"Using {len(search_path)} documents for processing.")

# %% [markdown]
# ###  Load spaCy language model

# %%
## DOWNLOAD SPACY MODEL and setup PIPELINE just once
# !uv run python3 -m spacy download en_core_web_trf
# nlp = spacy.load("en_core_web_trf")

# # Initialise spaCy pipeline 
#nlp.add_pipe("merge_entities")
# nlp.add_pipe("merge_noun_chunks")

# nlp.to_disk("./spacy_model_pipeline")


## RELOAD spacy pipeline
nlp = spacy.load("./spacy_model_pipeline")

# add CI_TYPE patterns to spacy nlp model pipeline
config = {"spans_key": None, "annotate_ents": True, "overwrite": False}
## see for more info: https://spacy.io/usage/rule-based-matching#entityruler
## NOTE EntityRuler is hidden inside .add_pipe()
try:
    ruler = nlp.add_pipe("span_ruler", config=config)
    ruler.from_disk(s.NER_PATTERNS_FILEPATH)
except ValueError:
    print("SpanRuler already exists in pipeline.")
    ruler = nlp.get_pipe("span_ruler")
    ruler.from_disk(s.NER_PATTERNS_FILEPATH)


# %%


# %% [markdown]
# ## Document cleaning
# 

# %%
# # Docling pipeline configs

# accelerator_options = AcceleratorOptions(
#     num_threads=4, device=AcceleratorDevice.AUTO
# )  # use GPU + multi-threading
# pipeline_options = PdfPipelineOptions()
# pipeline_options.do_ocr = True
# pipeline_options.do_table_structure = (
#     True  # identify tables as such just not to have them in the TextItems later
# )
# pipeline_options.accelerator_options = accelerator_options
# pipeline_options.force_backend_text = True




# # setup converter for PDF and markdown
# converter = DocumentConverter(
#     allowed_formats=[InputFormat.PDF, InputFormat.MD],
#     format_options={
#         InputFormat.PDF: FormatOption(
#             pipeline_cls=StandardPdfPipeline,
#             pipeline_options=pipeline_options,
#             backend=PyPdfiumDocumentBackend,
#         ),
#     },
# )


# %%
# print( "Number of documents to process:", len(os.listdir(DOCS_DIR)) )
# start_time = time.time()


# # convert the different layouts of the pdf files into unified markdown format incl. sub/section titles, tables, caption text etc
# for pdf_filename in os.listdir(DOCS_DIR):

#     if pdf_filename.endswith(".pdf"):

#         md_filename = f"{Path(pdf_filename).stem}.md"

#         pdf_filepath = os.path.join(DOCS_DIR, Path(pdf_filename))
#         md_filepath = os.path.join(PARSED_TEXT_DIR, Path(md_filename))
#         cleaned_md_filepath = md_filepath.replace(".md", "_cleaned.md")

#         if os.path.exists(cleaned_md_filepath):
#             print(f"Cleaned markdown file already exists: '{cleaned_md_filepath}'")
#             continue

#         print(f"\nFetching: {pdf_filename}")

#         print("Remove reference section")
#         pdf_text = extract_text(pdf_filepath)
#         pdf_text_no_refs = dc.remove_references(pdf_text)

#         print("Removing URLs") # LangExtract tries to open these URLs when they occur in the document text
#         pdf_text_no_refs_urls = re.sub(r"http\S+", "", pdf_text_no_refs) 

#         # FIXME remove workaround of saving pdf as markdown and reading it again as Docling.Document
#         with open(md_filepath, "w", encoding="utf-8") as f:
#             f.write(pdf_text_no_refs_urls)
#         print("Converting Markdown to text...")
#         # FIXME with DocLoader
#         #loader = DoclingLoader(md_filepath)
#         # md_text = loader.load()
#         md_text = converter.convert(md_filepath)

#         print("Removing headers and footers\n")
#         md_text_cleaned = dc.remove_headers_footers(md_text)

#         ## Translation
#         src_language_doc = langdetect.detect(pdf_filename.replace(".pdf", "").lower())  # lower case improves language detection

#         if src_language_doc != "en":
#             supported_languages = ["fr", "de", "es", "it", "itc", "nl"]
#             if src_language_doc not in supported_languages:
#                 print(f"Unsupported source language: {src_language_doc}. Continue with extraction on original text")
#                 continue 
            
#             print(f"Translating {src_language_doc} --> en")
#             md_text_cleaned.document.texts = tm.translate_2_english(src_language_doc, md_text_cleaned.document.texts)        


#         print(f"Saving parsed and cleaned document as markdown to: {cleaned_md_filepath}")
#         md_text_cleaned.document.save_as_markdown(cleaned_md_filepath)


# end_time = time.time() - start_time
# print(f"Parsing and cleaning done. Time elapsed: {end_time:.2f} seconds.")


# # visual check of removed items
# # TODO make as document_cleaning function: print removed items with largest number of chars first
# # ## NOTE. high number of chars == more potentially actual text body

# # text_items_removed = sorted(text_items_to_drop_visualization, key=lambda x: -x[0])
# # for i in text_items_removed[:50]:
# #     print(i) # -->  also subsection titles were removed partly





# %% [markdown]
# 

# %%


# %% [markdown]
# ## LLama

# %% [markdown]
# ###  Prompt engineering

# %%
question_1 = "Which infrastructure failures are mentioned in the text? Categorize the output by the type of infrastructure, the location, and the type of damage."

# %%



# %% [markdown]
# ### Init Llama application function 
# * extract CI-GEO pairs (ie. most likely geolocation of each CI_TYPE )
# * pass NER table to prompt for evaluating and improving LLM response
# * Test approach by applying it on three cleaned documents
# 

# %% [markdown]
# ### Load GeoLlama


# %%


# %% [markdown]
# ### Apply Llama on chunks

# %%

gc.collect()
torch.cuda.empty_cache() 
torch.no_grad()
print(torch.cuda.memory_reserved() / 1e9)


# %%


# %% [markdown]

# %%
## Settings

model_name = "meta-llama/Llama-2-7b-chat-hf"
# TODO NOTE test for HPC:   need > 2 GPUs
# model_name = "meta-llama/Llama-3.1-8B-Instruct"


time0 = time.time()

## init LLM pipeline
decoder_model = em.DecoderModel(model_name)


## init GeoLLM pipeline
## Make sure that still both GPUS are visible
# print(os.environ["CUDA_VISIBLE_DEVICES"])
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# print(os.environ["CUDA_VISIBLE_DEVICES"])
# !nvidia-smi


topo_model = TopoModel(
    model_name='JoeShingleton/GeoLlama-3.2-3b-toponym',
    # model_name='JoeShingleton/GeoLlama_7b_toponym', 
    prompt_path='./submodules/geollama/data/prompt_templates/prompt_template.txt',
    instruct_path='./submodules/geollama/data/prompt_templates/topo_instruction.txt',
    input_path=None,
    config_path='./submodules/geollama/data/config_files/model_config.json'
)

rag_model = RAGModel(
    model_name='JoeShingleton/GeoLlama-3.2-3b-RAG',
    # model_name='JoeShingleton/GeoLlama_7b_RAG',
    prompt_path='./submodules/geollama/data/prompt_templates/prompt_template.txt',
    instruct_path='./submodules/geollama/data/prompt_templates/rag_instruction.txt',
    input_path='./submodules/geollama/data/prompt_templates/rag_input.txt',
    config_path='./submodules/geollama/data/config_files/model_config.json')

geo_llama = GeoLlama(
    topo_model = topo_model, 
    rag_model = rag_model, 
    translate_model=None
)

gc.collect()
torch.cuda.empty_cache() 
torch.no_grad()
print(torch.cuda.memory_reserved() / 1e9)


# CI-GEO pairs
geolocs_cache = geonamescache.GeonamesCache()
countries = geolocs_cache.get_countries()
ci_geo_countries = [*u.gen_dict_extract(countries, 'name')]



## init outputs
df_responses_all_step1 = pd.DataFrame()
df_responses_all_step2 = pd.DataFrame()
sentence_root_list = [] # for visualization of sentence roots and their dependencies
responses_error_list = []
df_geollm_response = pd.DataFrame()


if test_mode:
    search_path = docs_list_sample
    print("Test mode is ON. Using only a small sample of documents for testing.")
else:
    search_path = glob(str(Path(PARSED_TEXT_DIR, "*cleaned.md")))


## Start CI impact extraction
for i, filename in enumerate(search_path):

    src_language_nonengl = None 

    no_documents = len(search_path)
    filepath = Path(filename)
    filename_stem = filepath.stem



    ## load doc
    loader = DoclingLoader(filepath)  # use chunks from Docling.Loader
    doc = loader.load()


    print(f"\n\n ######## -------- Processing document [{i+1}/{no_documents}]: {filepath.name} -------- ######## \n")

    ## extract authors, publication year and title 
    author, year, title = dc.extract_citation_info(filename_stem)
    citation = f"{author} {year}".replace("  ", " ").strip()
    title = title.replace(" - ", "").replace("_cleaned", "").strip()



    print(f"\n ######## -------- Relationship extraction: CI-GEO pairs -------- ######## \n")

    ## Create New entity for transport infrastructure and apply it on any doc

    ## TODO make as pydantic class with fixed attributes
    df_ci_geo = pd.DataFrame(
        columns=[
            "citation_id",
            "chunk_id",
            "ci_entity",
            "geo_entity",
            "case_type",
            "chunk_text",
            "token_distance",
        ]
    )

    print(f"\n ######## -------- Getting geolocation of CI assets -------- ######## \n")
    for chunk_no, chunk in enumerate(doc):

        ## preprocess  TODO move to document cleaning workflow + dc.funcs
        chunk.page_content = chunk.page_content.replace("\n", " ")
        chunk.page_content = chunk.page_content.replace("- ", "-") # TODO test if ("- ", "") is better


        ## get most likely geolocation for each CI entity based on distance between tokens
        nlp_chunk = nlp(chunk.page_content)
        all_ents = [ent for ent in nlp_chunk.ents]
        ci_type_ents = [ent for ent in nlp_chunk.ents if ent.label_ in ["CI_TYPE", "FAC"]]


        # check if chunk contains CI_TYPE entities
        if len(ci_type_ents) > 0:

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

                                ## check that GPE,LOC are not countries (too coarse info CI-GEO pair)
                                ## of GPE/LOC is country -> proceed with next GPE/LOC 
                                if all_ents[ent_idx].text in ci_geo_countries:
                                    continue

                                geo_idx = ent_idx
                                dist_ent_pair = np.abs(ci_idx - geo_idx)
                                distance_list.append(dist_ent_pair)
                                idx_in_chunk.append((ent_idx))
                                closest_pair_idx = np.argmin(
                                    distance_list
                                )  # idx of closest GEO entity
                                distance_closest_pair = distance_list[closest_pair_idx]

                        threshold = 5  # max token distance between CI_TYPE and GEO entity
                        if distance_closest_pair > threshold:
                            # print(
                            #     f""" Token distance is too large between CI_TYPE/FAR and next GEO entity which is of {distance_closest_pair} [token distance] > {threshold} [max. token distance] """
                            # )
                            continue
                        else:
                            pass

                        ## write as dict entry incl chunk_id, ci_entity, geo_entity, distance
                        result_dict = {
                            "chunk_id": chunk_no,
                            "chunk_text": chunk.page_content,
                            "ci_entity": all_ents[ci_idx].text,
                            "ci_entity_label": all_ents[ci_idx].label_,
                            "geo_entity": all_ents[idx_in_chunk[closest_pair_idx]].text,
                            "geo_entity_label": all_ents[idx_in_chunk[closest_pair_idx]].label_,
                            "token_distance": distance_closest_pair,
                        }
                        df_ci_geo = pd.concat(
                            [df_ci_geo, pd.DataFrame([result_dict])], ignore_index=True
                        )

                    except IndexError:
                        continue
        else:
            # print("\nNo CI_TYPE or FAC entities found in this chunk.")
            continue

    
    ## post-process of DF CI-GEO pairs for each chunk
    unique_ci_geo_pairs = df_ci_geo.drop_duplicates(
        subset=["citation_id", "chunk_id", "ci_entity", "geo_entity","case_type", "chunk_text"])
    print("number of duplicates to remove:", len(df_ci_geo) - len(unique_ci_geo_pairs))

    df_ci_geo = df_ci_geo.drop_duplicates(
        subset=["citation_id", "chunk_id", "ci_entity", "geo_entity","case_type", "chunk_text"]
        )# .reset_index(drop=True, inplace=True)



    print(f"\n  #############  -------- Text-2-Data: {filepath.name} -------- #############  \n")

    df_responses_step1 = pd.DataFrame(
        columns=[
            "citation_id",
            "chunk_id",
            "infrastructure_type",
            "damage",
            "location",
            "ci_entity",
            "geo_entity",
            "case_type",
            "chunk_text"
        ]
    )

    df_responses_step2 = pd.DataFrame(
        columns=[
            "citation_id",
            "chunk_id",
            "infrastructure_type",
            "infrastructure_group",
            "damage",
            "location",
            "latitude",
            "longitude",
            "ci_entity",
            "geo_entity",
            "case_type",
            "coord_potential_locations",
            "chunk_text"
        ]
    )
    df_geollm_response = pd.DataFrame()

    
    ## apply decoder on each chunk in document
    ## TODO replace iteration by loading entire document and use recursive chunking from langchain
    for j, chunk in enumerate(doc):
        
        print("\n\nCHUNK NO.", j)
        print("Chunk text: ", chunk.page_content)

        # clean up before applying CUDA
        gc.collect()
        torch.cuda.empty_cache() 
        torch.no_grad()
        # print(torch.cuda.memory_reserved() / 1e9)
    
        ## Start geoparsing with GeoLLM to verify Location
        # load GeoLLM prompt template
        template_geollm = em.load_prompt_template(template_filename="geollm.txt")
        
        
        # extract locations
        # for d in tqdm(chunk.page_content):
        resp = geo_llama.geoparse(chunk.page_content)
        # TODO add here geoLLm prompt as var
        # TODO check if useful in model.py to set : model.use_checkpointing = True or  model.gradient_checkpointing_enable()
        
        # save results
        df_geollm_resp = pd.DataFrame(resp[0:])
        df_geollm_resp["chunk_text"] = chunk.page_content
        
        df_geollm_response = pd.concat([df_geollm_response, df_geollm_resp], ignore_index=True)
        
        try:
            locations_per_chunk = df_geollm_resp["name"].to_list()
            lats_per_chunk = df_geollm_resp["latitude"].to_list()
            lons_per_chunk = df_geollm_resp["longitude"].to_list()
            RAGestimated_per_chunk = df_geollm_resp["RAG_estimated"].to_list()
        except:
            print("No locations extracted by GeoLLM for this chunk. Going to next chunk\n", resp[0:])
            continue
        
        # sanity check that only cases with location infos are considered for further processing      
        assert len(locations_per_chunk) > 0, "No potential locations found for this chunk."

        ## Input for LLM 1  -1st round
        if df_ci_geo.loc[df_ci_geo["chunk_id"] == j].empty:

            context = [
                {
                    "text": chunk.page_content,
                    "citation": citation,
                    "title": filename_stem,
                    "entity_recognition": None,
                },
            ]

        else:  # TODO dissolve if else clause by making it in ci_locations: if df_ci_geo.chunk=j, xx, else None
            context = [
                {
                    "text": chunk.page_content,
                    "citation": citation,
                    "title": filename_stem,
                    "entity_recognition": df_ci_geo.loc[df_ci_geo['chunk_id'] == j],
                },
            ]

        # load LLM_1 prompt template
        # template_1 = em.load_prompt_template(template_filename="ci_loc_direct_impacts_org.txt")
        template_1 = em.load_prompt_template(template_filename="ci_loc_direct_impacts_geollm_step_1.txt")

        # apply LLM 1
        response = decoder_model.generate_response(
            question=question_1, context=context, # chunk_id=j
            prompt_template = template_1,
            temperature = 0.2,
            max_new_tokens = 1024
        )
        
        ## postprocess response
        try:
            df_resp = pp.postprocess_response(response[0]["generated_text"])

            # save LLM response for each chunk  in dataframe for each document
            df_resp["citation_id"] = context[0]["citation"]  # add citation info
            df_resp["chunk_id"] = j  # add chunk id as identifier
            df_resp["ci_entity"] = df_ci_geo.loc[df_ci_geo["chunk_id"] == j, "ci_entity"] 
            df_resp["geo_entity"] = df_ci_geo.loc[df_ci_geo["chunk_id"] == j, "geo_entity"]
            df_resp["case_type"] = df_ci_geo.loc[df_ci_geo["chunk_id"] == j, "case_type"]
            df_resp["chunk_text"] =  context[0]["text"]  # add (translated) chunk text for tracing back LLM response
        
            df_responses_step1 = pd.concat([df_responses_step1, df_resp], ignore_index=True)
            print("df_resp 1")
            print(df_resp)

        except (IndexError, ValueError) as e:
            print(f"Cannot add response: {e},\nFaulty response (before postprocessing):", response[0]['generated_text'].replace('\n', ''))
            # faulty response: e.g.  .., "location": "V" on satellite and online on radar"}, { ...}, {}
            responses_error_list.append({
                "citation_id": citation,
                "chunk_id": j,
                "response": response[0]['generated_text'].replace('\n', ''),
                "error": str(e)
            })
            print("Continue with next chunk\n")
            continue



        # load NER patterns for CI types and their subgroups
        ci_patterns = pd.read_json("./ner_patterns.jsonl/patterns", lines=True)
        
        ## group Ci types into subgroups,
        df_resp = pp.group_ci_types(df_resp, "infrastructure_type", "infrastructure_group", ci_patterns)
        ## keep only records which are actually about CI (e.g., not theatre, stadion ..)
        df_resp.dropna(subset=["infrastructure_group"], inplace=True)
    

        ## Input for LLM - 2nd round (GeoLLM)
        # print(f"\nCHECKING input.text, df_resp, GeoLLm resp \n{chunk.page_content}\n{df_resp},\n{locations_per_chunk}")

        template_step2 = em.load_prompt_template(template_filename="ci_loc_direct_impacts_geollm_step_2.txt")

        context = [
            {
                "text": chunk.page_content,
                "citation": citation,
                "title": filename_stem, 
                "previous_response": df_resp,
                "potential_locations": locations_per_chunk,  # list of 1 or multiple location strings           
                "coordinates_potential_locations": list(zip(locations_per_chunk, lats_per_chunk, lons_per_chunk, RAGestimated_per_chunk)),
            },
        ]
        

        response = decoder_model.generate_response(
            question=question_1, context=context, # chunk_id=j
            prompt_template = template_step2,
            temperature = 0.2,
            max_new_tokens = 1024
        )
                
        ## postprocess response
        try:
            df_resp = pp.postprocess_response(response[0]["generated_text"])
        
            # save LLM response for each chunk  in dataframe for each document
            df_resp["citation_id"] = context[0]["citation"]  # add citation info
            df_resp["chunk_id"] = j  # add chunk id as identifier
            df_resp["ci_entity"] = df_ci_geo.loc[df_ci_geo["chunk_id"] == j, "ci_entity"] 
            df_resp["geo_entity"] = df_ci_geo.loc[df_ci_geo["chunk_id"] == j, "geo_entity"]
            df_resp["case_type"] = df_ci_geo.loc[df_ci_geo["chunk_id"] == j, "case_type"]
            df_resp["coord_potential_locations"] = list(zip(locations_per_chunk, lats_per_chunk, lons_per_chunk, RAGestimated_per_chunk)), # INTERIM for verification of lat, lon 
            df_resp["chunk_text"] =  context[0]["text"]  # add (translated) chunk text for tracing back LLM response
        
            # collect resps for each doc
            df_responses_step2 = pd.concat([df_responses_step2, df_resp], ignore_index=True) # 
            print("df resp 2")
            print(df_resp)

        except (IndexError, ValueError) as e:
            print(f"Cannot add response: {e},\nFaulty response (before postprocessing):", response[0]['generated_text'].replace('\n', ''))
            # faulty response: e.g.  .., "location": "V" on satellite and online on radar"}, { ...}, {}
            responses_error_list.append({
                "citation_id": citation,
                "chunk_id": j,
                "response": response[0]['generated_text'].replace('\n', ''),
                "error": str(e)
            })

    df_responses_all_step1 = pd.concat([df_responses_all_step1, df_responses_step1], ignore_index=True)        
    df_responses_all_step2 = pd.concat([df_responses_all_step2, df_responses_step2], ignore_index=True)

    # clean up after each document
    gc.collect()
    torch.cuda.empty_cache()  # mainly needed after training, small effect when LLM applied only for inference
    torch.no_grad()


print(f"\n\n ######## -------- CI impact extraction took {(time.time() - time0) / 60} minutes -------- ######## \n\n")


# %% [markdown]
# 1. FIX CI hallucis: rerun LLm with only CI-geo pairs (no geollm) and org.workflow and prompt (fix cigeo-dublicates)
# 
# * currently-local-fixedGROUP_CI check if old WF (oldWF vs newWF) is better than this one here (newWF) - similar scores in CI recall:0.5, f1:0.14-0.17
# <!-- 
# # ### NEW WF
# 
# <!-- ### OLD WF
# 
# # 
# (maybe solved via geonamescache) --> LLM-step1 summarizes loc too much (belgium, instead of Orte --> soll idese als einzl dict rausschrieben 
# 
# #### --> test OUTP als list od dic mit ci1, loc1, ci2, loc2 ...) falls org LLM-ci-geo zu selben problem führt
# 
# currently 2. re-add GeoLLM  (step2)  (+ prompt to recheck CI correct link to loc)
# 
# ### 3. fix location coords e.g. by selecting "location" from retunred list_of_potential_locs
# 
# ### 4. Doc cleaning (rm Abstracts, and footers Nat.Haz. ..)
# 
# 
# 

# %%



# %%


# %% [markdown]
# 

# %% [markdown]
# ### FIX. check geollm ~ llm_1 interaction

# %%

print(df_responses_step2.infrastructure_group.isna().sum())  # mostly cases which are not CI (theater, stadion..)
print(df_responses_step2.infrastructure_group.value_counts()) # four most common subgroups seems to be correct
# df_pred.infrastructure_group.unique()



# %%
df_responses_step2
# df_responses_all_step2
# df_responses_step2.loc[20:30, ["infrastructure_type", "infrastructure_group", "damage", "location", "chunk_text"]]


# %%

gc.collect()
torch.cuda.empty_cache() 
torch.no_grad()
print(torch.cuda.memory_reserved() / 1e9)


# %%
print("Chunk with erroneous responses:", responses_error_list.__len__())
df_responses_error = pd.DataFrame(responses_error_list)
df_responses_error#.tail(3)


# %%
df_responses_all_step2#.tail(3)

# %% [markdown]
# ### Saving

# %%
# PATH_LLM_DATA: Path = Path(s.PATH_DATA /"llm_outputs/")
# LLM_DATA_FILENAME: str = "llm_1_updprompt_distanceNER.csv"

# OUTPUT_LLM1_FILEPATH =  Path(PATH_LLM_DATA / LLM_DATA_FILENAME)#.replace(".csv", "_v2.csv"))
# OUTPUT_LLM1_FILEPATH 
df_responses_step1.to_csv("llm1_geollm_step1_partly.csv", index=False)
df_responses_step2.to_csv("llm1_geollm_step2_partly.csv", index=False)


# %%
safety_df = df_responses_all_step2.copy()



# save LLM 1 output to disk along with prompt text
if not os.path.isfile(OUTPUT_LLM1_FILEPATH):

    print(f"Saving prompt, LLM response and erroneous responses [.txt, .csv] to {OUTPUT_LLM1_FILEPATH} ...")

    with open(OUTPUT_LLM1_FILEPATH.parent / f"prompt_{OUTPUT_LLM1_FILEPATH.stem}.txt", "w") as f:
        f.write(template_1.render(context=context, question=question_1))
    df_responses_all_step2.to_csv(OUTPUT_LLM1_FILEPATH, index=False)
    df_responses_error.to_csv(OUTPUT_LLM1_FILEPATH.parent / f"errors_{OUTPUT_LLM1_FILEPATH.stem}.csv", index=False)

elif os.path.isfile(OUTPUT_LLM1_FILEPATH) and not os.path.isfile(OUTPUT_LLM1_FILEPATH.parent / f"{OUTPUT_LLM1_FILEPATH.stem}_v2.csv"):

    print(f"Output file {Path(OUTPUT_LLM1_FILEPATH).stem} already exists. Saving as {OUTPUT_LLM1_FILEPATH.stem}_v2 to avoid overwriting ...")

    # If the original files exists but the v2 file doesn't, create the v2 file
    with open(OUTPUT_LLM1_FILEPATH.parent / f"prompt_{OUTPUT_LLM1_FILEPATH.stem}_v2.txt", "w") as f:
        f.write(template_1.render(context=context, question=question_1))
    df_responses_all_step2.to_csv(Path(OUTPUT_LLM1_FILEPATH.parent, f"{OUTPUT_LLM1_FILEPATH.stem}_v2.csv"), index=False)
    df_responses_error.to_csv(Path(OUTPUT_LLM1_FILEPATH.parent / f"errors_{OUTPUT_LLM1_FILEPATH.stem}_v2.csv"), index=False)

else:
    print(f"Output file {Path(OUTPUT_LLM1_FILEPATH).stem} already exists. Skip saving to avoid overwriting ...")






# %% [markdown]
# Save as pyarrow incl dtypes as pyarraws-. Saving also dtypes a spyarrow objectes leads to quicker loading and processing

# %%
# pd.options.future.infer_string = True

df_responses_all_pyarrow = df_responses_all_step2.astype( dtype="string[pyarrow]")
df_responses_all_pyarrow = pa.Table.from_pandas(df_responses_all_pyarrow)
pq.write_table(df_responses_all_pyarrow, OUTPUT_LLM1_FILEPATH.with_suffix(".parquet"))  



# %%
response

# %%
# df_ci_geo
j = 3
df_resp["ci_entity"] = df_ci_geo.loc[df_ci_geo["chunk_id"] == j, "ci_entity"]
df_resp.sort_values(by="ci_entity")

# %%
# 12 docs with fixed NERpatterns: 74min

# %%
print(torch.cuda.memory_reserved() / 1e9)


# %%
print("Chunk with erroneous responses:", responses_error_list.__len__())
df_responses_error = pd.DataFrame(responses_error_list)
df_responses_error#.tail(3)


# %%
df_responses_all_step2#.tail(3)

# %%




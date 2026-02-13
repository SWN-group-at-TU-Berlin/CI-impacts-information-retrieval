# %% [markdown]
# ## Aim:
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
import os
import sys
import subprocess
import importlib

import re
import time
from glob import glob
from pathlib import Path

from io import StringIO

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from langchain_docling import DoclingLoader
import langdetect

import spacy
from huggingface_hub import login
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)

import torch

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


sys.path.append("../")
from src.settings import settings as s
import src.document_cleaning as dc
import src.translation_model as tm
# from src.hf_model_access import get_weight_dir


torch.manual_seed(42)

#  automatic linebreaks and multi-line cells.
pd.set_option("display.max_colwidth", None)
pd.set_option("display.colheader_justify", "left")


test_mode = True

torch.cuda.empty_cache() 
print(torch.cuda.memory_reserved() / 1e9)


# %%
# os.environ["HF_HOME"] = s.HF_HOME_DIR
try: 
    login(token=os.getenv("HUGGINGFACE_TOKEN"))   # notebook_login
except:
    login(token=os.environ.get("HUGGINGFACE_TOKEN"))  # former HF_TOKEN


test_mode = True # for testing purposes, limit number of processed documents



# %% [markdown]
# ## Set paths and vars

# %%
# set wd to project root
os.chdir("/home/a-buch/Documents/TUB_DWN/_PROJECTS/CI-impacts-information-retrieval")



## set path variables
DOCS_DIR = Path(s.PATH_DATA + "text_sources/")
PARSED_TEXT_DIR = Path(s.PATH_DATA + "parsed_documents/")
LLM_OUTPUTS_DIR = Path(s.PATH_DATA + "llm_outputs/")
NER_PATTERNS_FILEPATH = Path(s.NER_PATTERNS_FILEPATH)

# document cleaning
md_dir = Path(PARSED_TEXT_DIR)
md_dir.mkdir(parents=True, exist_ok=True)

# CI GEO pairs
CI_GEO_FILEPATH = Path("./" + s.PATH_DATA + s.CI_GEO_PAIRS_FILENAME)

## store LLM 1 response and prompt
os.makedirs(s.PATH_LLM_DATA, exist_ok=True)
OUTPUT_LLM1_FILEPATH =  Path(s.PATH_LLM_DATA / s.LLM_DATA_FILENAME)
OUTPUT_PROMPT_FILEPATH = Path(s.PATH_LLM_DATA / f"prompt_{OUTPUT_LLM1_FILEPATH.stem}.txt" )




# %% [markdown]
# ### Set test mode

# %%

if test_mode:
    print("Test mode is ON. Using only a small sample of documents for testing.")

    docs_list_sample = [
        Path(PARSED_TEXT_DIR, "AEMET 2024 - ESTUDIO SOBRE LA SITUACIÓN DE LLUVIAS INTENSAS_cleaned.md"),
        Path(PARSED_TEXT_DIR, "Karakatsani 2023 - Greece economy briefing The economic impact of the recent devastating floods in Greece_cleaned.md"),
        #Path(PARSED_TEXT_DIR, "Koks 2022 - Brief communication_cleaned.md"),
        # Path(PARSED_TEXT_DIR, "Khazai 2013 - Juni-Hochwasser 2013 in Mitteleuropa - Fokus Deutschland Bericht 2 Auswirkungen und Bewältigung_cleaned.md"),
        # Path(PARSED_TEXT_DIR, "European Investment Bank 2025 - Spain_ EIB lends €50 million to Iberdrola to rebuild and climate-proof flood-hit power infrastructure in Valencia_cleaned.md"),
        # Path(PARSED_TEXT_DIR, "Wilson 2024 - Flash floods in Spain sweep away cars, disrupt trains and leave several missing _ AP News_cleaned.md"),     
        # Path(PARSED_TEXT_DIR, "Wildhagen 2013 - Hochwasser_ Wie die Flut Unternehmen lahmlegt_cleaned.md"),
        # Path(PARSED_TEXT_DIR, "Lloyd's List 2024 - Port of Valencia reopens after devastating floods_cleaned.md"),
        # Path(PARSED_TEXT_DIR, "Containerlift 2024 - Valencia Port Resumes Operations Following Devastating Flooding in Spain - Containerlift.co.uk - Transport_Lifting_Shipping_cleaned.md"), 
    ]


## Test mode
if test_mode:
    search_path = docs_list_sample
    print("Test mode is ON. Using only a small sample of documents for testing.")
else:
    search_path = glob(str(Path(PARSED_TEXT_DIR, "*cleaned.md")))


# %% [markdown]
# ###  Load and initialize spaCy language model

# %%
# # spaCy language model for NER and ENTITY LINKING

try: 
    print("Try loading spaCy language model for local machine ...")
    try:
        nlp = spacy.load(s.SPACY_MODEL)
    except (OSError, ValueError):
        print(f"spaCy language model '{s.SPACY_MODEL}' not found. Downloading ...")
        ## loading transformer language model for NER requires additional package
        if (s.SPACY_MODEL.endswith("_trf")):
            !uv add spacy[transformers]
        !uv run python -m spacy download {s.SPACY_MODEL}
        nlp = spacy.load(s.SPACY_MODEL)

except (OSError, ValueError):
    try: 
        print(f"spaCy language model '{s.SPACY_MODEL}' not found. Downloading ...")
        subprocess.check_call(["uv", "pip", "install", "spacy-transformers"])
        subprocess.check_call(["uv", "run", "python3", "-m", "spacy", "download", s.SPACY_MODEL])
        nlp = spacy.load(s.SPACY_MODEL)
    except ValueError:
        print("Try downloading directly via python3 -m spacy ... (and not via uv run ...)")
        subprocess.check_call(["python3", "-m", "spacy", "download", s.SPACY_MODEL])


# !uv run python -m spacy download en_core_web_trf
# nlp = spacy.load("en_core_web_trf")

# %%
# Initialise spaCy pipeline 
nlp.add_pipe("merge_entities")
nlp.add_pipe("merge_noun_chunks")


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


# %%
## Document cleaning

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
        pdf_text_no_refs = dc.remove_references(pdf_text)

        print("Removing URLs") # LangExtract tries to open these URLs when they occur in the document text
        pdf_text_no_refs_urls = re.sub(r"http\S+", "", pdf_text_no_refs) 

        # FIXME remove workaround of saving pdf as markdown and reading it again as Docling.Document
        # FIXME with DocLoader
        # loader = DoclingLoader(md_filepath)
        # md_text = loader.load()
        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(pdf_text_no_refs_urls)
        print("Converting Markdown to text...")
        md_text = converted.convert(md_filepath)

        print("Removing headers and footers\n")
        md_text_cleaned = dc.remove_headers_footers(md_text)

        print(f"Saving parsed and cleaned document as markdown to: {cleaned_md_filepath}")
        md_text_cleaned.document.save_as_markdown(cleaned_md_filepath)

        end_time = time.time() - start_time
        print(f"Parsing and cleaning done. Time elapsed: {end_time:.2f} seconds.")


# visual check of removed items
# TODO make as document_cleaning function: print removed items with largest number of chars first
# ## NOTE. high number of chars == more potentially actual text body

# text_items_removed = sorted(text_items_to_drop_visualization, key=lambda x: -x[0])
# for i in text_items_removed[:50]:
#     print(i) # -->  also subsection titles were removed partly





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
env = Environment(loader=FileSystemLoader("./prompt_templates/"))
template = env.get_template("ci_loc_direct_impacts.txt")


# %% [markdown]
# ### Init Llama application function 
# * extract CI-GEO pairs (ie. most likely geolocation of each CI_TYPE )
# * pass NER table to prompt for evaluating and improving LLM response
# * Test approach by applying it on three cleaned documents
# 

# %%
# # empty CUDA cache
import gc
import torch

gc.collect()

torch.cuda.empty_cache()
torch.no_grad()
# print(torch.cuda.memory_summary(device=None, abbreviated=False))



# %%
# init class for decoder and tokenizer


class DecoderModel:

    def __init__(self, model_name: str ="meta-llama/Llama-2-7b-chat-hf"):
        
        try: 
            login(token=os.getenv("HUGGINGFACE_TOKEN"))   # notebook_login
        except:
            login(token=os.environ.get("HUGGINGFACE_TOKEN"))  # former HF_TOKEN

        base_dir =  s.HF_HOME_DIR   # use default dir in .cache/
        model_dir = base_dir # / f"models--{model_name.replace("/", "--")}"  # is already .._mirror/hub/
        
        print(model_dir)

        # quantization config
        # Load model with 4-bit quantization if applicable (use 4-bit integer instead of 32b floats) --> reduce the required VRAM for model application
        # see, https://huggingface.co/docs/transformers/quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.pipeline, self.tokenizer = self.initialize_model(
            model_name, model_dir, bnb_config
        )
        

    def initialize_model(self, model_name: str, model_dir: str = None, bnb_config=None):
        
        # Model and Tokenizer initialization
        if not os.path.exists(model_dir):
            print("Model directory not found. Downloading model...")
            os.makedirs(model_dir, exist_ok=True)

            device = transformers.infer_device()
            print(f"Using device: {device}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype="auto",
                attn_implementation="flash_attention_2",  # use with 4-bit quantization,
                # --> flash attention enables to use much larger sequence lengths without running into OOM issues
                quantization_config=bnb_config,
                # max_memory={0: "2GB", 1: "10GB"},  # distribute memory across GPUs
            )
            model.save_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            tokenizer.save_pretrained(model_dir)

            print("Downloaded model and tokenizer")

        else:
            print(f"Using locally saved model from {model_dir}")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=model_dir,
                local_files_only=True,  # tp_plan="auto" # set tensor parallel model (ie. splits model on multiple GPU)
                dtype="auto",
                attn_implementation="flash_attention_2",  # use with 4-bit quantization,
                # --> flash attention enables to use much larger sequence lengths without running into OOM issues
                quantization_config=bnb_config,
                # tp_plan="auto",  # automatically use a tensor parallelism plan based on predefined configuration of the model (i.e. partition model on both GPUs)
            )
            # print("Tensor parallel plan:", model._tp_plan)

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                cache_dir=model_dir,  # use fast Rust-based tokenizer, when possible
            )

        # reduce further memory usage
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.use_checkpointing = True

        torch.cuda.empty_cache()
        torch.no_grad()

        # Pipeline setup for question answering
        pipeline = transformers.pipeline(  # load model locally from wsl .cache\
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024, # high max token otherwise output is truncated
            device_map="auto",
        )
        return pipeline, tokenizer

    def generate_response(
        self, question: str, context: list, # chunk_id: int
    ):
    
        rendered_prompt = template.render(
            context=context,  # includes also df_ci_geo info
            question=question,
        )

        # print(f"Generating response for chunk_id: {chunk_id} ...")

        sequences = self.pipeline(
            rendered_prompt,  # jinja template
            max_new_tokens=1024, # use default to not truncate the LLM response
            do_sample=True,
            num_beams=1,  # select token based on probability distribution over entire model’s vocabulary
            # top_k=10,
            # top_p=0.5,
            temperature=0.1,
            # num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,  # allow bullet point answers
        )
        # Extracting and returning the generated text
        return sequences

# %% [markdown]
# ### Apply Llama on chunks

# %%
## Settings

model_name = "meta-llama/Llama-2-7b-chat-hf"
# test for HPC: "meta-llama/Llama-3.1-8B-Instruct"



## init LLM pipeline
decoder_model = DecoderModel(model_name=model_name)

## init output dataframe
df_responses_all = pd.DataFrame()
sentence_root_list = [] # for visualization of sentence roots and their dependencies
responses_error_list = []

test_mode = True
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


    # check language of document based on document title
    # check title rather than chunk_text as later might contain also footer or header info in another languages
    
    src_language_doc = langdetect.detect(str(title).lower())  # lower case improves language detection
    
    if src_language_doc != "en":
        supported_languages = ["fr", "de", "es", "it", "itc", "nl"]
        if src_language_doc not in supported_languages:
            # raise ValueError(f"Unsupported source language: {src_language_doc}")
            print(f"Unsupported source language: {src_language_doc}. Continue with extraction on original text")
            continue 
        
        print(f"\n ######## -------- Translating: {src_language_doc} --> en -------- ######## \n")

        # clean up before applying translator
        gc.collect()
        torch.cuda.empty_cache()  # mainyl after training needed, small effect when LLM applied only for infernece
        torch.no_grad()
        
        # return translated Docling Document
        doc = tm.translate_2_english(src_language_doc, doc)



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
        ]
    )

    for chunk_no, chunk in enumerate(doc):


        ## apply customized NLP pipeline -> returns same chunk.text but with CI_entities annotations and without linebreak symbols
        for nlp_chunk in nlp.pipe([chunk.page_content]):

            ci_entities = [ent for ent in nlp_chunk.ents if ent.label_ in ["CI_TYPE"]]

            # check if chunk has no CI_TYPE entities
            if not ci_entities:
                print(f"Chunk [{chunk_no}], CI_TYPE entities found: 0")
                # print(f"\nChunk [{chunk_no}], No. CI_TYPE and FAC entities: {len(ci_entities)}")
                continue
      

            print(f"Chunk [{chunk_no}], CI_TYPE entities found: {len(ci_entities)}")
            

            for ci_ent in ci_entities:
                # print("Ci_ent:", ci_ent.text)

                # Get the root of the sentence 
                sentence_root = ci_ent.sent.root

                if sentence_root.dep_ in ("ROOT", "dobj", "conj"):  # TODO probably enough when set only to ROOT 
                    # sentence_root_list.append(sentence_root.dep_) # for visualization of sentence roots and their dependencies
                    head = ci_ent.root.head
                    # print(f" `{sentence_root, sentence_root.dep_}` is sentence_root, \n `{head}` the direct head of ci entity `{ci_ent}`")

                    subjs = list(sentence_root.lefts) + list(sentence_root.rights)
                    for subj in subjs:
                        for descendant in subj.subtree:

                            # check if descendant is a location entity
                            if descendant.ent_type_ in ("GPE", "LOC"):
                                descendant_loc = descendant
                                print("  ", ci_ent.text, "->", descendant_loc.text)
                                # assert subj is descendant_loc or subj.is_ancestor(descendant_loc)

                                ## Special case handling - one root but multiple ci-geo pairs
                                try:
                                    first_ci_ancestor = [ancestor.text for ancestor in descendant_loc.ancestors if ancestor.ent_type_ in ("CI_TYPE", "FAC")][0]
                                    if ci_ent.text == first_ci_ancestor:
                                        print(
                                            "descendant of location:", descendant_loc.text, descendant_loc.dep_, 
                                            "\nits ancestors:", [ancestor.text for ancestor in descendant_loc.ancestors] # all parents up to root.head
                                        )
                                        print("  ", ci_ent.text, "-->", descendant_loc.text)
                                        
                                        ## write as dict entry incl chunk_id, ci_entity, geo_entity, distance
                                        result_dict = {
                                            "citation_id": citation,
                                            "chunk_id": chunk_no,
                                            "ci_entity": ci_ent.text,
                                            "geo_entity": descendant_loc.text,
                                            "case_type": "special",
                                            "chunk_text": nlp_chunk.text,
                                        }
                                        df_ci_geo = pd.concat(
                                            [df_ci_geo, pd.DataFrame([result_dict])], ignore_index=True
                                        )

                                ## usual case handling
                                except IndexError:
                                    ## write as dict entry incl chunk_id, ci_entity, geo_entity, distance
                                    result_dict = {
                                        "citation_id": citation,
                                        "chunk_id": chunk_no,
                                        "ci_entity": ci_ent.text,
                                        "geo_entity": descendant_loc.text,
                                        "case_type": "usual",
                                        "chunk_text": nlp_chunk.text,
                                    }
                                    df_ci_geo = pd.concat(
                                        [df_ci_geo, pd.DataFrame([result_dict])], ignore_index=True
                                    )

                else:
                    # print("\nNo CI_TYPE or FAC entities found in this chunk.")
                    continue

    ## post-process of DF CI-GEO pairs
    df_ci_geo = df_ci_geo.drop_duplicates(
        subset=["citation_id", "chunk_id", "ci_entity", "geo_entity","case_type", "chunk_text"]
        )# .reset_index(drop=True, inplace=True)


    print(f"\n  #############  -------- Text-2-Data: {filepath.name} -------- #############  \n")

    df_responses = pd.DataFrame(
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

    ## apply decoder on each chunk in document
    ## TODO replace iteration by loading entire document and use recursive chunking from langchain
    for j, chunk in enumerate(doc):

        if df_ci_geo.loc[df_ci_geo["chunk_id"] == j].empty:
            context = [
                {
                    "text": chunk.page_content,
                    "citation": citation,
                    "title": filename_stem, 
                    "ci_locations": None,
                },
            ]
        else:  # TODO disolve if else clause by making it in ci_locations: if df_ci_geo.chunk=j, xx, else None
            context = [
                        {
                            "text": chunk.page_content,
                            "citation": citation,
                            "title": filename_stem, 
                            # TODO check if chunk id is unique per doc and if multiple ci_loc pairs eixist (then just tke first one)
                            "ci_locations": df_ci_geo.loc[df_ci_geo["chunk_id"] == j],
                        },
                    ]

        # apply LLM
        response = decoder_model.generate_response(
            question=question_1, context=context, # chunk_id=j
        )

        ## postprocess response
        resp = response[0]["generated_text"].replace("\n", "")
        try:
            #  remove potential text outside of json object
            resp = (resp.split("]")[0] + "]") 
            resp = ("[" + resp.split("[")[1]) 
            # fix missing bracket at end of response
            if "]" not in resp:
                resp = resp.rpartition('}')[-3] + "}]"

            # save LLM response for each chunk  in dataframe for each document
            df_resp = pd.read_json(StringIO(resp))
            df_resp["citation_id"] = context[0]["citation"]  # add citation info
            df_resp["chunk_id"] = j  # add chunk id as identifier
            df_resp["ci_entity"] = df_ci_geo.loc[df_ci_geo["chunk_id"] == j, "ci_entity"] or None
            df_resp["geo_entity"] = df_ci_geo.loc[df_ci_geo["chunk_id"] == j, "geo_entity"] or None
            df_resp["case_type"] = df_ci_geo.loc[df_ci_geo["chunk_id"] == j, "case_type"]
            df_resp["chunk_text"] = context[0]["text"]  # add (translated) chunk text for tracing back LLM response

            df_responses = pd.concat([df_responses, df_resp], ignore_index=True)
        
        except ValueError as e:
            print(f"Cannot add response: {e}, \n   Response (before postprocessing): {response[0]['generated_text'].replace('\n', '')}")
            responses_error_list.append({
                "citation_id": citation,
                "chunk_id": j,
                "response": response[0]['generated_text'].replace('\n', ''),
                "error": str(e)
            })


    df_responses_all = pd.concat([df_responses_all, df_responses], ignore_index=True)


    # clean up after each document
    gc.collect()
    torch.cuda.empty_cache()  # mainly needed after training, small effect when LLM applied only for inference
    torch.no_grad()

# %%
print("Chunk with erroneous responses:", responses_error_list.__len__())
df_responses_error = pd.DataFrame(responses_error_list)
df_responses_error.tail(3)

# %%


# %% [markdown]
# #### response

# %%
# OUTPUT_LLM1_FILEPATH.name.replace(".csv", "_v2.csv")


# %%
safety_df = df_responses_all.copy()

# OUTPUT_LLM1_FILEPATH =  Path(s.PATH_LLM_DATA / s.LLM_DATA_FILENAME.replace(".csv", "_v2.csv"))


# save LLM 1 output to disk along with prompt text
if not os.path.isfile(OUTPUT_LLM1_FILEPATH):

    print(f"Saving prompt, LLM response {OUTPUT_LLM1_FILEPATH.stem}, and erroneous LLM responses [.txt, .csv] to {OUTPUT_LLM1_FILEPATH.parent} ...")
    
    with open(OUTPUT_PROMPT_FILEPATH, "w") as f:
        f.write(template.render(context=context, question=question_1))
    
    df_responses_all.to_csv(OUTPUT_LLM1_FILEPATH, index=False)
    df_responses_error.to_csv(Path(OUTPUT_LLM1_FILEPATH.parent, f"errors_{OUTPUT_LLM1_FILEPATH.stem}.csv"), index=False)

else:
    print(f"Output file {Path(OUTPUT_LLM1_FILEPATH).stem} already exists. Skip saving to avoid overwriting ...")



    # # check if file exists already
    # if os.path.exists(OUTPUT_LMM1_FILEPATH):
    #     print(f"File {OUTPUT_LMM1_FILEPATH.name} already exists. Not saving to disk.")
    #     pass
    # else:
    #     print(f"Saving LLM 1 output to {OUTPUT_LMM1_FILEPATH}")
    #     df_responses_all.to_csv(OUTPUT_LMM1_FILEPATH, index=False)

    




# %% [markdown]
# 

# %%


# %% [markdown]
# ## Evaluation - only manually

# %% [markdown]
# ### Manual comparison CI_location_table vs LLm response
# 

# %% [markdown]
# #### chunk 5
# In Germany, road and railway infrastructure was severely damaged as documented exemplarily in Fig. 1. Cost estimates reach up to EURO 2 billion Euro (MDR, 2021). More than 130 km of motorways were closed directly after the event, of which 50 km were still closed two months later, with an estimated repair cost of EUR 100 million (Hauser, 2021). Of the 112 bridges in the ﬂooded 40 km of the Ahr valley (Rhineland-Palatinate), 62 bridges were destroyed, 13 were severely damaged and only 35 were in operation a month after the ﬂood event (MDR, 2021). Over 74 km of roads, paths and bridges in the Ahr valley have been (critically) damaged. In some cases, repairs are expected to take months to years (Zeit Online, 2021). For example, major freeway sections, including parts of the A1 motorway, were closed until early 2022 (24Rhein, 2022). In addition, about 50 000 cars were damaged, causing insurance claims of some EUR 450 million (ADAC, 2021). The German railway provider Deutsche Bahn expects asset damages of around EUR 1.3 billion. Among other things, 180 level crossings, almost 40 signal'

# %%
doc[5].page_content

# %%
df_ci_geo[df_ci_geo["chunk_id"] == 5]

# %%
df_responses[df_responses["chunk_id"] == 5]

# %%
print(
    f"CI_TYPE \n LLM responses:\n {df_responses.infrastructure_type.unique()}, \n\n NER pairs:\n {df_ci_geo.ci_entity.unique()}"
)

# %%
print(
    f"LOCATION:\n LLm responses:\n {df_responses.location.unique()}, \n\n NER pairs:\n {df_ci_geo.geo_entity.unique()}"
)

# %% [markdown]
# ### chunk 13 +14
# 'We found no information regarding direct impact on solid-waste facilities as a result of the ﬂood event. However, there is a large pressure on the solid-waste sector to clean the affected areas; 1 month after the event, we observed dozens of large temporary waste ﬁlls and frequent incidences of oil pollution in Rhineland-Palatinate during a ﬁeld visit. In the Ahrweiler district alone, the ﬂood caused as much solid waste as normally would be collected over 30 years. In Belgium, the amount of solid waste is estimated around 160 000 t, stored at several places, such as the abandoned highway track A601. This highway has been used for approximately 9 months as a temporary storage for debris (Couplez, 2022). In the Netherlands, there have been primarily problems with waste deposits along the river banks, which is mostly the solid waste transported by the river from further upstream. Thousands of tonnes of tree debris (logs and\nNat. Hazards Earth Syst. Sci., 22, 3831–3838, 2022\nE. E. Koks et al.: Flood impacts to infrastructure'
# 
# 
# 'of running water and electricity (Ärzte Zeitung, 2021). After 1.5 months, medical care was guaranteed again in the most affected regions in Rhineland-Palatinate (Hochwasser Ahr, 2021c). In the state of North Rhine-Westphalia, approximately 68 hospitals have been affected, of which several have been affected severely and will take at least 1.5 years to be rebuilt (Fig. 2). Direct damages are estimated to be at least EUR 100 million to repair all medical facilities (Korzilius, 2021). In the town of Eschweiler (Germany), for example, the basement of the hospital was ﬂooded, as well as the outbuildings and the entire outdoor area. The power supply collapsed, the entire building technology was destroyed and some 300 patients had to be evacuated by helicopter. Property damage is expected to be around EUR 50 million. Within 3.5 weeks, the hospital was partly operational, and within 3 months, all hospital operations continued normally (SAH Eschweiler, 2021). The Mutterhaus Ehrang hospital in Trier (Germany) is now permanently closed as the hospital is too severely damaged to rebuild. Furthermore, in the region of Rhineland-Palatinate (Germany), 19'
# 

# %%
doc[14].page_content

# %%
df_ci_geo[df_ci_geo["chunk_id"].isin([13, 14])]

# %%
df_responses[df_responses["chunk_id"].isin([13, 14])]

# %%
print(
    f"CI_TYPE \n LLM responses:\n {df_responses.infrastructure_type.unique()}, \n\n NER pairs:\n {df_ci_geo.ci_entity.unique()}"
)

# %%
print(
    f"LOCATION:\n LLm responses:\n {df_responses.location.unique()}, \n\n NER pairs:\n {df_ci_geo.geo_entity.unique()}"
)

# %% [markdown]
# ## Test alternative approaches for entity linking / relation extraction

# %%
import torch
import gc

print(torch.cuda.memory_summary(device=None, abbreviated=False))
# # empyty CUDA cache
gc.collect()

torch.cuda.empty_cache()
torch.no_grad()
# print(torch.cuda.memory_summary(device=None, abbreviated=False))

# %%




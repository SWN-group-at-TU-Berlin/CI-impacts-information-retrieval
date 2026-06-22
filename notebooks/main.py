# %%


# %%
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

#
import os

# # settings for CUDA and PYTORCH
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# print(os.environ["CUDA_VISIBLE_DEVICES"])
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["PYTORCH_ALLOC_CONF"]="expandable_segments:True" ## improve memory allocation

# # settings for debugging CUDA errors (pinpoint exact line of error)
os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 

# activate global venv explicitly
os.environ["VIRTUAL_ENV"] = "/beegfs/home/users/a/a-buch/_PROJECTS/CI-impacts-information-retrieval/.venv"


import torch

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())  # should give 2
# print(torch.cuda.get_device_name())
# print(torch.cuda.get_device_properties(0))
# # print(torch.cuda.get_device_properties(1))
# print(torch.cuda.get_device_capability())
# print(torch.cuda.get_arch_list())
# print(torch.__version__)
# print(torch.version.cuda)


## --> must be CUDA 12.6, torch: 2.91, ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']

import os
import sys
# import subprocess
import re
import time
from glob import glob
from pathlib import Path
import gc
from typing import List, Dict, Tuple, Optional, Union
# from io import StringIO
# import json

# from tqdm import tqdm
import numpy as np
import pandas as pd
import geonamescache
# import pyarrow as pa
# import pyarrow.parquet as pq
import spacy
from huggingface_hub import login
from pdfminer.high_level import extract_text
import langdetect
from transformers import AutoTokenizer
from haystack.dataclasses import ByteStream
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types import DoclingDocument
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, DocItem
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
# from langchain.document_loaders import DirectoryLoader #, UnstructuredLoader
# from langchain_community.document_loaders import DirectoryLoader, UnstructuredLoader
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.document_converter import DocumentConverter, FormatOption
from docling.chunking import HybridChunker


sys.path.append("./")
from src.settings import settings as s
import src.document_cleaning as dc
import src.translation_model as tm
import src.extraction_model as em
import src.postprocess as pp
import src.utils as u
from geollama.geollama.main import GeoLlama
from geollama.geollama.model import TopoModel, RAGModel




test_mode = True

# login to HF
# login(token="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")#os.getenv("HUGGINGFACE_TOKEN"))
try: 
    login(token=os.getenv("HUGGINGFACE_TOKEN"))   # notebook_login
except:
    try:
        login(token="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")#os.getenv("HUGGINGFACE_TOKEN"))
    except:
        login(token=os.environ.get("HUGGINGFACE_TOKEN"))  # former HF_TOKEN
        

# NOTE raises exception if not env.variable doesnt exist (compared to os.envrion.get and its shortcut os.getenv)


# NOTE. disabled batch size as OOM for CUDA despite chunkwise memory cleaning, nvtop to find best batchsize
BATCH_SIZE = s.BATCH_SIZE  # max for nvidia GPU

# torch.manual_seed(42)

#  automatic linebreaks and multi-line cells.
pd.set_option('display.max_colwidth', 100000)
pd.set_option("display.colheader_justify", "left")


# clean up before applying CUDA
gc.collect()
torch.cuda.empty_cache() 
print(torch.cuda.memory_reserved() / 1e9)
torch.no_grad()



# ## TODO test to prevent CUDA-OOM when reused
# ## Source: https://spacy.io/usage/embeddings-transformers
# from thinc.api import set_gpu_allocator, require_gpu

# # Use the GPU, with memory allocations directed via PyTorch.
# # This prevents out-of-memory errors that would otherwise occur from competing
# # memory pools.
# set_gpu_allocator("pytorch")
## require_gpu(0)



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
        # Path(PARSED_TEXT_DIR, "Lloyd's List 2024 - Port of Valencia reopens after devastating floods.md"),
        Path(PARSED_TEXT_DIR, "Containerlift 2024 - Valencia Port Resumes Operations Following Devastating Flooding in Spain - Containerlift.co.uk - Transport_Lifting_Shipping.md"), 
        Path(PARSED_TEXT_DIR, "ABC 2024 - Traffic jams and flight delays due to heavy rain and lightning storm in Malaga.md"),

        Path(PARSED_TEXT_DIR, "Karakatsani 2023 - Greece economy briefing The economic impact of the recent devastating floods in Greece.md"),
        Path(PARSED_TEXT_DIR, "European Investment Bank 2025 - Spain_ EIB lends €50 million to Iberdrola to rebuild and climate-proof flood-hit power infrastructure in Valencia.md"),
        Path(PARSED_TEXT_DIR, "Wilson 2024 - Flash floods in Spain sweep away cars, disrupt trains and leave several missing _ AP News.md"),     
            Path(PARSED_TEXT_DIR, "Wildhagen 2013 - Hochwasser_ Wie die Flut Unternehmen lahmlegt.md"),
            Path(PARSED_TEXT_DIR, "AFP 2022 - The_Vibes_Valencia Airport in Madrid briefly shut as lightning hits runway _ World _ The Vibes.md"),
            Path(PARSED_TEXT_DIR, "Artemis 2015 - PERILS finalises Storm Desmond UK flood loss estimate at £604m.md"),
            Path(PARSED_TEXT_DIR, "Brown 2010 - Economy feels chill as UK grinds to a halt _ The Independent.md"),
        #     # Path(PARSED_TEXT_DIR, "Diakakis 2020 - A systematic assessment of the effects of extreme flash floods on transportation infrastructure and circulation: The example of the 2017 Mandra flood.md"),
        Path(PARSED_TEXT_DIR, "EFE 2024 - The DANA storm, live_ The death toll rises to 158.md"),
        #     # Path(PARSED_TEXT_DIR, "Eurelectric 2006 - Impacts of Severe Storms on Electric Grids.md"),
            Path(PARSED_TEXT_DIR, "Euronews 2024 - Spain floods_ Death toll rises to 205 as nation braces for more rain .md"),
        Path(PARSED_TEXT_DIR, "Ferlita 2023 - Incendi in Sicilia, ecco cosa accade.md"),
            Path(PARSED_TEXT_DIR, "Fink 2004 - The 2003 European summer heatwaves and drought - synoptic diagnosis and impacts.md"),
        Path(PARSED_TEXT_DIR, "Gilbody Dickerson 2024 - Spain floods_ At least 95 people killed including British man near Malaga _ World News _ Sky News.md"),
        #     # Path(PARSED_TEXT_DIR, "Kadir 2014 - The Impact of Natural Disasters on Critical Infrastructures - A Domino Effect-based Study.md"),
            Path(PARSED_TEXT_DIR, "Kaur 2025 - Authorities suspect arson in 17 wildfires across Dalmatian coast, Croatia - The Watchers.md"),
        #     # Path(PARSED_TEXT_DIR, "Keller 2014 - Mapping Natural Hazard Impacts on Road Infrastructure—The Extreme Precipitation in Baden-Württemberg.md"),
            Path(PARSED_TEXT_DIR, "Kettle 2020 - Storm Xaver over Europe in December 2013 Overview of energy impacts and North Sea events.md"),
        #     # Path(PARSED_TEXT_DIR, "Koks 2019 - Understanding Business Disruption and Economic Losses Due to Electricity Failures and Flooding.md"),
        #     # Path(PARSED_TEXT_DIR, "Korzilius 2021 Nach der Flut.md"),

    Path(PARSED_TEXT_DIR, "Khazai 2013 - Juni-Hochwasser 2013 in Mitteleuropa - Fokus Deutschland Bericht 2 Auswirkungen und Bewältigung.md"),
    Path(PARSED_TEXT_DIR, "Rozendaal 2021 - Infrabel_ Flood damage to railway track worth tens of millions of euros _ SpoorPro - incomplete.md"),
    Path(PARSED_TEXT_DIR, "Skoulding 2023 - Where are the fires in Italy today as temperatures rise to 47.6C on Sicily_ _ The Independent.md"),
    Path(PARSED_TEXT_DIR, "Treanor 2015 - Storm Desmond damage across Cumbria estimated at £500m _ Storm Desmond _The Guardian.md"),

    # # long processing
    # Path(PARSED_TEXT_DIR, "AEMET 2024 - ESTUDIO SOBRE LA SITUACIÓN DE LLUVIAS INTENSAS.md"),
    # Path(PARSED_TEXT_DIR, "Koks 2022 - Brief communication.md"),
    
        #     # not part of valid set:
        #     # Path(PARSED_TEXT_DIR, "Krausmann 2014 - STREST report on lessons learned from recent catastrophic events.md"), # > 1800 entries LLMv3.0 incl. hallucinations
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


# load NER patterns for CI types and their subgroups (needed for cleaning LLm response - STEP 1 before continuing with STEP 2)
ci_patterns = pd.read_json("./ner_patterns.jsonl/patterns", lines=True)

# %%


# %% [markdown]
# ### geollama pipeline
# 
# 

# %%
## Make sure that still both GPUS are visible

# print(os.environ["CUDA_VISIBLE_DEVICES"])
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# print(os.environ["CUDA_VISIBLE_DEVICES"])
# # !nvidia-smi




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



# %%


# %% [markdown]
# #### loader - bruise code 

# %%
##########################  HYPHEN CLENAING SOLO  (kkep solo or inc. in DoclingParser class)

## document-wise cleaning 

import re

# noinspection PyPackageRequirements
import nltk
from haystack.dataclasses import ByteStream
from docling_core.types import DoclingDocument
from docling_core.types.doc import CoordOrigin
from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, DocItem


# Module-level caches (private)
_words_list = None
_lemmatizer = None
_stemmer = None


# ## load nltk libs for handling hyphens
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def get_words_list():
    """Lazily load and cache the NLTK english words list."""
    global _words_list
    if _words_list is None:
        nltk.download('words')
        _words_list = set(nltk.corpus.words.words())
    return _words_list


def get_lemmatizer():
    """Lazily load and cache the WordNetLemmatizer."""
    global _lemmatizer
    if _lemmatizer is None:
        from nltk.stem import WordNetLemmatizer
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def get_stemmer():
    """Lazily load and cache the PorterStemmer."""
    global _stemmer
    if _stemmer is None:
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
    return _stemmer


def is_valid_word(word):
    """
    Check if a word is valid by comparing it directly and via stemming/lemmatization.
    In detail, it checks if the given word, its stem, or its lemma is inlcuded in the word list downloaded from nltk or the customized list of suffixes.

    Returns True (or the valid modified word) if the word is found,
    otherwise returns False.
    """
    words_list = get_words_list()
    stemmer = get_stemmer()
    lemmatizer = get_lemmatizer()

    stem = stemmer.stem(word)
    if word.lower() in words_list or word in words_list:
        return True
    elif stem in words_list or stem.lower() in words_list:
        return True

    # Check all lemmatizations of the word
    for pos in ['n', 'v', 'a', 'r', 's']:
        lemma = lemmatizer.lemmatize(word, pos=pos)
        if lemma in words_list:
            return True

    # Check for custom lemmatizations
    # noinspection SpellCheckingInspection
    suffixes = {
        "ability": "able",  # testability -> testable
        "ibility": "ible",  # possibility -> possible
        "iness": "y",  # happiness -> happy
        "ity": "e",  # creativity -> create
        "tion": "e",  # creation -> create
        "able": "",  # testable -> test
        "ible": "",  # possible -> poss
        "ing": "",  # running -> run
        "ed": "",  # tested -> test
        "s": ""  # tests -> test
    }
    for suffix, replacement in suffixes.items():
        if word.endswith(suffix):
            stripped_word = word[:-len(suffix)] + replacement
            # Recursively check the modified word; if valid, return the modified form.
            result = is_valid_word(stripped_word)
            if result:
                return result

    return False


def combine_hyphenated_words(p_str):
    """
    Combine hyphenated words if the parts together form a valid word.
    Otherwise, preserve the hyphen (assuming it connects two valid words).
    """

    def replace_dash(match):
        word1, word2 = match.group(1), match.group(2)
        combined = word1.strip() + word2.strip()

        # If there is a space after the hyphen and the combined word is valid,
        # assume the hyphen was splitting a single word.
        if word2.startswith(" ") and is_valid_word(combined):
            return combined
        # If both parts are valid words on their own, keep them hyphenated.
        elif is_valid_word(word1.strip()) and is_valid_word(word2.strip()):
            return word1.strip() + '-' + word2.strip()
        # Otherwise, if the combined word is valid, return it.
        elif is_valid_word(combined):
            return combined
        # If the combined word starts with a capital letter (likely a proper noun)
        # and the second part isn’t valid on its own, combine them.
        elif combined[0].isupper() and not word2.strip()[0].isupper() and not is_valid_word(word2.strip()):
            return combined

        # Default: assume the hyphen is meant to connect two words.
        return word1.strip() + '-' + word2.strip()

    # Replace any soft hyphen characters with a regular dash.
    p_str = p_str.replace("­", "-")
    # Look for hyphens between word parts (with or without an extra space)
    p_str = re.sub(r'(\w+)-(\s?\w+)', replace_dash, p_str)

    return p_str

# %%
### taken and adapted from https://github.com/brucenielson/BookSearchArchive/blob/e2d6c4145d7931648d5854ba29186cbec8150e87/docling_parser.py
### and related blog post: https://www.mindfiretechnology.com/blog/archive/finding-paragraphs-in-pdfs-using-ibm-s-docling/

from typing import List, Dict, Tuple, Optional, Union
from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, DocItem


def clean_text(p_str: str) -> str:
    p_str = str(p_str).strip()  # Convert text to a string and remove leading/trailing whitespace
    p_str = p_str.encode('utf-8').decode('utf-8')
    p_str = re.sub(r'\s+', ' ', p_str).strip()  # Replace multiple whitespace with single space
    p_str = re.sub(r"([.!?]) '", r"\1'", p_str)  # Remove the space between punctuation (.!?) and '
    p_str = re.sub(r'([.!?]) "', r'\1"', p_str)  # Remove the space between punctuation (.!?) and "
    p_str = re.sub(r'\s+\)', ')', p_str)  # Remove whitespace before a closing parenthesis
    p_str = re.sub(r'\s+]', ']', p_str)  # Remove whitespace before a closing square bracket
    p_str = re.sub(r'\s+}', '}', p_str)  # Remove whitespace before a closing curly brace
    p_str = re.sub(r'\s+,', ',', p_str)  # Remove whitespace before a comma
    p_str = re.sub(r'\(\s+', '(', p_str)  # Remove whitespace after an opening parenthesis
    p_str = re.sub(r'\[\s+', '[', p_str)  # Remove whitespace after an opening square bracket
    p_str = re.sub(r'\{\s+', '{', p_str)  # Remove whitespace after an opening curly brace
    p_str = re.sub(r'(?<=\s)\.([a-zA-Z])', r'\1',
                   p_str)  # Remove a period that follows a whitespace and comes before a letter
    p_str = re.sub(r'\s+\.', '.', p_str)  # Remove any whitespace before a period

    # Remove footnote numbers at end of a sentence. Check for a digit at the end and drop it
    # until there are no more digits or the sentence is now a valid end of a sentence.
    while p_str and p_str[-1].isdigit() and not is_sentence_end(p_str):
        p_str = p_str[:-1].strip()
    
    return p_str

##########################
def remove_figure_references(p_str: str) -> str:
    # remove potneital figure reference when they are colsed by bracketss, e.g. (A1), (B20)
    # this is done to avoid mismatches with road names
    p_str = re.sub(r"\s+\([A-Z][0-9]{1,}\)\s", "", p_str)
    return p_str

def is_reference_section(document_text: str) -> bool:
    # search for reference section
    pattern = re.compile(
        r"^(References|REFERENCES|Bibliography|BIBLIOGRAPHY)$", flags=re.MULTILINE
    )
    # re.MULTILINE in combination with "^" and case sensitive : find search words only when they are at beginning of a new line
    matches = re.findall(pattern, document_text)
    if matches:
        print(f"Reference section found!" )
        return True
###################



def is_sentence_end(text: str) -> bool:
    has_end_punctuation: bool = is_ends_with_punctuation(text)
    # Does it end with a closing bracket, quote, etc.?
    ends_with_bracket: bool = (text.endswith(")")
                               or text.endswith("]")
                               or text.endswith("}")
                               or text.endswith("\"")
                               or text.endswith("\'"))
    return (has_end_punctuation or
            (ends_with_bracket and is_ends_with_punctuation(text[0:-1])))


def combine_paragraphs(p1_str: str, p2_str: str):
    # If the paragraph ends without final punctuation, combine it with the next paragraph
    if is_sentence_end(p1_str):
        return p1_str + "\n" + p2_str
    else:
        return p1_str + " " + p2_str



def is_section_header(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    if text is None:
        return False
    return text.label == "section_header"


def is_page_footer(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label == "page_footer"


def is_page_header(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label == "page_header"


def is_footnote(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label == "footnote"


def is_list_item(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label == "list_item"


def is_text_break(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return is_page_header(text) or is_section_header(text) or is_footnote(text)


def is_page_not_text(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label not in ["text", "list_item", "formula"]


def is_page_text(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return not is_page_not_text(text)


def is_ends_with_punctuation(text: str) -> bool:
    return text.endswith(".") or text.endswith("?") or text.endswith("!")


def is_too_short(doc_item: DocItem, threshold: int = 2) -> bool:
    return doc_item.label == "text" and len(doc_item.text) <= threshold


def is_bottom_note(text: DocItem, near_bottom: bool = False) -> bool:
    # if 'Morgenstern was then the director' in text.text:
    #     pass
    # if text.text.startswith("10. Summing up o f"):
    #     pass

    # If it is specifically digits followed by a period, followed by a space, and it is
    # a section header or a list item, then it is NOT a bottom note
    if bool(re.match(r"^\d+\.\s", text.text)) and (is_section_header(text) or is_list_item(text)):
        return False
    # If it's digits followed by a letter without a space then it's a bottom note
    if bool(re.match(r"^\d+[A-Za-z]", text.text)):
        return True

    if text is None or not is_page_text(text):
        return False
    # Check for · at the beginning of the line. This is often how OCR represents footnote number.
    if text.text.startswith("·") and not text.text.startswith("· "):
        return True

    if re.match(r"^\d", text.text):
        # If the first digit is zero, it can't be a footnote because that should never happen.
        if text.text.startswith("0"):
            return False
        if near_bottom:
            # Check if this is three digits with the third digit being a 1 followed by a space
            # This is usually where the last 1 was supposed to be an 'I'.
            return re.match(r"^\d{1,2}1 ", text.text) or not is_list_item(text)

    return False


def is_near_bottom(doc_item: DocItem, same_page_items: [DocItem], threshold: float = 0.3) -> bool:
    """
    Determine if a DocItem is near the bottom of its page.

    Parameters:
    - doc_item: The DocItem object containing provenance data with 'bbox'.
    - doc: The DoclingDocument containing all DocItems.
    - threshold: Distance in points from the bottom to consider as 'near the bottom'.

    Returns:
    - True if the DocItem is within the threshold from the bottom, False otherwise.
    """
    # Check if the DocItem has provenance data with a bounding box
    if hasattr(doc_item.prov[0], 'bbox'):
        bbox = doc_item.prov[0].bbox
    else:
        return False  # No bounding box available

    # Extract the coordinate origin and bounding box coordinates
    coord_origin = bbox.coord_origin
    x0, y0, x1, y1 = bbox.l, bbox.b, bbox.r, bbox.t

    # Find the maximum y1 value on the page
    page_top: float = max(item.prov[0].bbox.t for item in same_page_items if hasattr(item.prov[0], 'bbox'))
    # Find the min y1 value on the page
    page_bottom: float = min(item.prov[0].bbox.b for item in same_page_items if hasattr(item.prov[0], 'bbox'))
    page_size: float = page_top - page_bottom
    # Threshold is page_bottom + (size of page * threshold amount) (i.e. % of page to be considered the 'bottom')
    bottom_threshold: float = page_bottom + (page_size * threshold)

    if coord_origin == CoordOrigin.BOTTOMLEFT:
        # In this system, y1 is the distance from the top of the paragraph to the bottom of the page
        return y1 <= bottom_threshold
    elif coord_origin == CoordOrigin.TOPLEFT:
        # In this system, y1 is the distance from the top of the paragraph to the top of the page
        return y1 >= bottom_threshold
    else:
        raise ValueError("Unknown coordinate origin.")



def is_text_item(item: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return not (is_section_header(item)
                or is_page_footer(item)
                or is_page_header(item)
                # or is_reference_section(item)
            )



def get_next_text(texts: List[Union[SectionHeaderItem, ListItem, TextItem]], i: int) \
        -> Optional[Union[ListItem, TextItem]]:
    # Seek through the list of texts to find the next text item using is_text_item
    # Should return None if no more text items are found
    for j in range(i + 1, len(texts)):
        if j < len(texts) and is_text_item(texts[j]):  # skips page headers/footers
            return texts[j]
    return None


def is_roman_numeral(s: str) -> bool:
    roman_numeral_pattern = r'(?i)^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(roman_numeral_pattern, s.strip()))

def should_skip_element(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return any([
        is_page_footer(text),
        is_page_header(text),
        is_roman_numeral(text.text)
    ])


def get_processed_texts(doc: DoclingDocument) -> List[DocItem]:
    """
    Processes the document's text items page by page, separating regular content from notes
    (footnotes and bottom notes), and returns a list of DocItems with notes at the end.
    """
    regular_texts: List[DocItem] = []
    notes: List[DocItem] = []
    processed_pages: set[int] = set()  # Keep track of processed pages
    reached_bottom_notes: bool = False
    same_page_items: List[DocItem] = []
    near_bottom: bool = False
    mislabeled: List[DocItem] = []

    for text_item in doc.texts:
        page_number = text_item.prov[0].page_no

        if page_number not in processed_pages:
            # On new page, so get all items on the current page
            same_page_items = [
                item for item in doc.texts if item.prov[0].page_no == page_number
            ]
            processed_pages.add(page_number)  # Mark the page as processed
            reached_bottom_notes = False

        if not reached_bottom_notes:
            near_bottom = is_near_bottom(text_item, same_page_items, threshold=0.5)

        if is_too_short(text_item):
            continue
        elif reached_bottom_notes or is_footnote(text_item):
            notes.append(text_item)
        elif is_bottom_note(text_item, near_bottom=near_bottom):
            notes.append(text_item)
            reached_bottom_notes = True
        else:
            regular_texts.append(text_item)

        # Check if the DocItem is a SectionHeaderItem. If so, turn it into a TextItem.
        if reached_bottom_notes and is_section_header(text_item):
            mislabeled.append(text_item)

    return regular_texts + notes


def add_paragraph(
    text: str,               
    # para_num: int, section: str, page: Optional[int], 
    docs: List[ByteStream], 
    # meta: List[Dict]
):
    docs.append(ByteStream(text.encode('utf-8')))
    # meta.append({
    #     **meta_data,
    #     # "paragraph_#": str(para_num),
    #     "section_name": section,
    #     "page_#": str(page)
    # })




# %% [markdown]
# ## Document cleaning

# %%


# %%
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



# %%

# # print( "Number of documents to process:", len(os.listdir(DOCS_DIR)) )
# print("Number of documents to process:", len(search_path) )
# start_time = time.time()


# EXPORT_TYPE = ExportType.DOC_CHUNKS
# mislabeled: List[DocItem] = []
# min_paragraph_size = 100  
# temp_docs: List[ByteStream] = []
# temp_meta: List[Dict[str, str]] = []
# i: int
# combined_paragraph: str = ""
# combined_chars: int = 0
# para_num: int = 0
# section_name: str = ""
# page_no: Optional[int] = None
# first_note: bool = False




# # convert the different layouts of the pdf files into unified markdown format incl. sub/section titles, tables, caption text etc

# for filename in os.listdir(DOCS_DIR):
# # # for pdf_filename in search_path:

#     if filename.endswith(".pdf"):

#         pdf_filepath = os.path.join(DOCS_DIR, Path(filename))

#         md_filename = f"{Path(filename).stem}.md"
#         md_filepath = os.path.join(PARSED_TEXT_DIR, Path(md_filename))
#         cleaned_jsonl_filepath = md_filepath.replace(".md", "_cleaned.jsonl")
#         cleaned_md_filepath = md_filepath.replace(".md", "_cleaned.md")

#         if os.path.exists(cleaned_jsonl_filepath):
#             print(f"Cleaned markdown file already exists: '{cleaned_md_filepath}'")
#             continue
        

#         print(f"\nFetching: {filename}")

#         # get language of document
#         src_language_doc = langdetect.detect(filename.replace(".pdf", "").lower())  # lower case improves language detection


#         ## Document converter with OCR
#         print("Using OCR for text extraction as it identifies section titles, footers/headers and pagenumbers as such, but reads in also figure text sometimes") 
#         # NOTE all other standard doclingConverter retunr section/headers etc as BODY not FURNITURE
#         # NOTE: partly reads in figure text and table text 
#         pdf_doc_org = dc.DocumentParser().ocr_converter.convert(source=pdf_filepath).document  ## !! recognizes section titles, footers/headers !! :D

#         ##  get only list of Doc.items
#         texts = get_processed_texts(pdf_doc_org) 

#         texts_clean = []
#         section_names = []
#         for i, text in enumerate(texts):
            

#  ### as Parser Doc class
#             # get next text only when it is not page header/footer
#             next_text = get_next_text(texts, i)
#             # page_no = get_current_page(text, combined_paragraph, page_no)


#             # Update section header if the element is a section header
#             # TODO: Need a stronger check on section headers that takes top of page into account, etc
#             if is_section_header(text) and text not in mislabeled:
#                 print("!!  Section header found:", text.text)
#                 section_name = text.text
#                 continue

#             if is_reference_section(section_name):
#                 print("Reference section found. Stopping further processing of document.")
#                 break  

#             if should_skip_element(text):
#                 continue
            
#             # clean from double whitespace, newlines, etc.
#             p_str = clean_text(text.text)

#             # clean from potential figure references
#             p_str = remove_figure_references(p_str)

#             ## replace e.g. and i.e. --> eg and ie to avoid sentence splits
#             p_str = re.sub(r"e\.g\.\s+", "eg ", p_str)
#             p_str = re.sub(r"i\.e\.\s+", "ie ", p_str)

#             # Removing URLs 
#             # LangExtract tries to open these URLs when they occur in the document text
#             # p_str= re.sub(r"http\S+", "", p_str) 

#             p_str_chars = len(p_str)

#             # If the paragraph does not end with final punctuation, accumulate it
#             if not is_sentence_end(p_str):
#                 combined_paragraph = combine_paragraphs(combined_paragraph, p_str)
#                 combined_chars += p_str_chars
#                 continue

#             # p_str ends with a sentence end; decide whether to process or accumulate it
#             total_chars = combined_chars + p_str_chars
#             if is_section_header(next_text):
#                 # Immediately process if the next text is a section header
#                 p_str = combine_paragraphs(combined_paragraph, p_str)
#                 combined_paragraph, combined_chars = "", 0
#             elif total_chars < min_paragraph_size:
#                 # Not enough characters accumulated yet; decide based on next_text
#                 if next_text is None or (not is_page_text(next_text) and is_sentence_end(p_str)):
#                     # End of document or next text item is not a text item and current paragraph ends with punctuation
#                     # Process the paragraph and reset the accumulator even though this is a short paragraph
#                     p_str = combine_paragraphs(combined_paragraph, p_str)
#                     combined_paragraph, combined_chars = "", 0
#                 else:
#                     # Combine with next paragraph
#                     combined_paragraph = combine_paragraphs(combined_paragraph, p_str)
#                     combined_chars = total_chars
#                     continue
#             else:
#                 # Sufficient characters: process the paragraph and reset the accumulator
#                 p_str = combine_paragraphs(combined_paragraph, p_str)
#                 combined_paragraph, combined_chars = "", 0

#             p_str = combine_hyphenated_words(p_str)
#             if p_str:  # Only add non-empty content
#                 para_num += 1
#                 add_paragraph(
#                     p_str, 
#                     #para_num, section_name, page_no, 
#                     temp_docs, 
#                     # temp_meta
#                 )
#                 page_no = None
            
#             print("\n-__ Paragraph #", para_num, "; Section:", section_name)

#             # print(p_str)
#             # print(temp_docs, temp_meta)

# ##### as CLASS DCOPAraser=
# ### translator class
#             ## Translation
#             if src_language_doc != "en":

#                 supported_languages = ["fr", "de", "es", "it", "itc", "nl"]
#                 if src_language_doc not in supported_languages:
#                     print(f"Unsupported source language: {src_language_doc}. Continue with extraction on original text")
#                     continue 

#                 print(f"Translating {src_language_doc} --> en")
#                 p_str = tm.translate_2_english(src_language_doc, p_str)     
                
#             # collect cleaned text + meta data per doc            
#             texts_clean.append(p_str)
#             section_names.append(section_name)


#         print("TEST Saving cleaned text as DoclingDocument to store text (p_str)+ meta (sectiontitle, page_no): ")
#         # Initialize new DoclingDoc
#         doclingdoc = DoclingDocument(schema_name="DoclingDocument",  version="1.0.0",name="My Custom Document")
        
#         # write text and structural elements (titles, sections, and paragraphs) to doc     
#         title_node = doclingdoc.add_title(text=f"{filename.replace('.pdf', '')}")
#         for text, section_name in zip(texts_clean, section_names):
#             print(section_name, ":", text)
#             section_node = doclingdoc.add_heading(text=section_name, level=1, parent=title_node)
#             doclingdoc.add_text(label=DocItemLabel.TEXT, text=text, parent=section_node)

#         # Save doclingDocument as MD
#         doclingdoc.save_as_markdown(cleaned_md_filepath)     


#        ## STEP 2: then do chunking with HybridChunker  TODO
#         # # NOTE maybe only a bit needed as already very good splits but maybe tokensizes need to be adapted 
#         ## TODO inlcude p_str always as doclingobj part or write it back to DoclingObject, (maybe with section_name as meta info)

#         # chunking
#         chunk_iter = chunker.chunk(dl_doc=doclingdoc)
#         chunks = list(chunk_iter)

#         # # apply contextualization
#         # ser_text = chunker.contextualize(chunk=chunk)
#         # ser_tokens = tokenizer.count_tokens(ser_text)
#         # print(f"chunker.contextualize(chunk) ({ser_tokens} tokens):\n{ser_text!r}")
#         # print()

#         break


# end_time = time.time() - start_time
# print(f"Parsing and cleaning done. Time elapsed: {end_time:.2f} seconds.")


# # visual check of removed items
# # TODO make as document_cleaning function: print removed items with largest number of chars first
# # ## NOTE. high number of chars == more potentially actual text body

# # text_items_removed = sorted(text_items_to_drop_visualization, key=lambda x: -x[0])
# # for i in text_items_removed[:50]:
# #     print(i) # -->  also subsection titles were removed partly



# %%


# %% [markdown]
# ### bump from transformers to Langchain + Memory passing
# 

# %%

# class Memory:
#     # Source. https://medium.com/@jagadeesan.ganesh/mastering-llm-ai-agents-building-and-using-ai-agents-in-python-with-real-world-use-cases-c578eb640e35
#     def __init__(self):
#         self.memory_store = []

#     def remember(self, interaction):
#         self.memory_store.append(interaction)

#     def recall(self):
#         return " ".join(self.memory_store)
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate

# model_name = "meta-llama/Llama-3.1-8B-Instruct"


# # prompt_template = "Tell me a {adjective} joke"
# # prompt = PromptTemplate(input_variables=["adjective"], template=prompt_template)
# model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 dtype="auto", # None ,# test for CU12.6, torch.29.1 #"auto",
#                 # max_memory={0: "2GB", 1: "10GB"},  # distribute memory across GPUs
#             )
# # chain = prompt | model #| StrOutputParser()

# # chain.invoke("your adjective here")


# # Define a simple prompt for the agent
# template = """
# You are an AI assistant with expertise in data analysis and automation. Answer the following question:
# Question: {question}
# """

# # Set up the prompt and LLM chain
# prompt = PromptTemplate(template=template, input_variables=["question"])
# chain = prompt | model | StrOutputParser() # LLMChain(prompt=prompt, llm=llm)

# # # # Example query
# # query = "What is the impact of AI in healthcare?"
# # response = chain.run(question=query)
# # print(f"Agent Response: {response}")


# from langchain.agents import create_agent

# agent = create_agent(
#     model=model_name,
#     model_provider="huggingface",
#     # tools=[get_weather],
#     system_prompt="You are a helpful assistant",
#     temperature=0.0,
#     max_tokens=1024,
# )

# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
# )
# print(result["messages"][-1].content_blocks)

# %% [markdown]
# 
# ##  CI impact extraction
# 

# %%
# %%
# Settings
model_name = "meta-llama/Llama-3.1-8B-Instruct"

time0 = time.time()

# print(os.environ["CUDA_VISIBLE_DEVICES"])
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# print(os.environ["CUDA_VISIBLE_DEVICES"])

# clean up before applying CUDA
gc.collect()
torch.cuda.empty_cache() 
print(torch.cuda.memory_reserved() / 1e9)
torch.no_grad()


# Questions
question_1 = "Which infrastructure failures are mentioned in the text? Categorize the output by the type of infrastructure, the location, and the type of damage."
question_2 = "Is the location of each affected or damaged critical infrastructure correctly identified?"


## init LLM extraction models
decoder_model_1 = em.DecoderModelCaching(
    model_name,
    em.load_prompt_template(template_filename="short_static_llama3_NER.txt",)
)

decoder_model_2 = em.DecoderModelCaching(
    model_name,
    em.load_prompt_template(template_filename="short_static_llama3_NER_geollm_step2.txt",)
)


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
df_geollama_response = pd.DataFrame()


if test_mode:
    search_path = docs_list_sample
    print("Test mode is ON. Using only a small sample of documents for testing.")
else:
    search_path = glob(str(Path(PARSED_TEXT_DIR, "*cleaned.jsonl")))




## Start CI impact extraction
for file_no, filename in enumerate(search_path):

    EXPORT_TYPE = ExportType.DOC_CHUNKS
    mislabeled: List[DocItem] = []
    min_paragraph_size = 50  
    temp_docs: List[ByteStream] = []
    temp_meta: List[Dict[str, str]] = []
    i: int
    combined_paragraph: str = ""
    combined_chars: int = 0
    para_num: int = 0
    section_name: str = ""
    page_no: Optional[int] = None
    first_note: bool = False


    time1 = time.time()

    src_language_nonengl = None 

    no_documents = len(search_path)
    filepath = Path(filename)
    filename_stem = filepath.stem


    print(f"\n\n ######## -------- Processing document [{file_no+1}/{no_documents}]: {filepath.name} -------- ######## \n")

    ## extract authors, publication year and title 
    author, year, title = dc.extract_citation_info(filename_stem)
    citation = f"{author} {year}".replace("  ", " ").strip()
    title = title.replace(" - ", "").replace("_cleaned", "").strip()



    # init dfs to store interim results for each doc
    df_resp_step1 = pd.DataFrame(
        columns=[
            "citation_id",
            "chunk_id",
            "infrastructure_type",
            "damage",
            "damage_value",
            "location",
            "ci_entity",
            "geo_entity",
            "chunk_text"
        ]
    )
    df_resp_step2 = pd.DataFrame(
        columns=[
            "citation_id",
            "chunk_id",
            "infrastructure_type",
            "infrastructure_group",
            "damage",
            "damage_value",
            "location",
            "ci_entity",
            "geo_entity",
            "coord_potential_locations",
            "chunk_text"
        ]
    )
    
    print(f"\n ##### ------- Cleaning document -----------########")

    pdf_filepath = os.path.join(DOCS_DIR, Path(filename_stem + ".pdf"))
    md_filename = filename_stem + ".md"
    md_filepath = os.path.join(PARSED_TEXT_DIR, Path(md_filename))
    cleaned_md_filepath = md_filepath.replace(".md", "_cleaned.md")
    # cleaned_jsonl_filepath = md_filepath.replace(".md", "_cleaned.jsonl")

    time_cleaning = time.time()

    if os.path.exists(cleaned_md_filepath):
        print(f"Cleaned markdown file already exists, loading file and proceeding with Ci impact extraction")
        # Load the existing cleaned markdown file
        doclingdoc = md_converter.convert(cleaned_md_filepath).document      

    else:
        print(f"Cleaned markdown file does not exist yet. Cleaning document: '{filename}'")

        # get language of document
        src_language_doc = langdetect.detect(filename_stem.lower())  # lower case improves language detection


        ## Document converter with OCR
        print("Using OCR for text extraction as it identifies section titles, footers/headers and pagenumbers as such, but reads in also figure text sometimes") 
        # NOTE all other standard doclingConverter retunr section/headers etc as BODY not FURNITURE
        # NOTE: partly reads in figure text and table text 
        pdf_doc_org = dc.DocumentParser().ocr_converter.convert(source=pdf_filepath).document

        ##  get only list of Doc.items
        texts = get_processed_texts(pdf_doc_org) 

        texts_clean = []
        section_names = []

        # text cleaning, annotating section names, remove header/footers, and translation
        for i, text in enumerate(texts):
            

    ### TODO make as Parser Doc class
            # get next text only when it is not page header/footer
            next_text = get_next_text(texts, i)
            # page_no = get_current_page(text, combined_paragraph, page_no)


            # Update section header if the element is a section header
            # TODO: Need a stronger check on section headers that takes top of page into account, etc
            if is_section_header(text) and text not in mislabeled:
                print("!!  Section header found:", text.text)
                section_name = text.text
                continue

            if is_reference_section(section_name):
                print("Reference section found. Stopping further processing of document.")
                break  

            if should_skip_element(text):
                continue
            
            # clean from double whitespace, newlines, etc.
            p_str = clean_text(text.text)

            # clean from potential figure references
            p_str = remove_figure_references(p_str)

            ## replace e.g. and i.e. --> eg and ie to avoid sentence splits
            p_str = re.sub(r"e\.g\.\s+", "eg ", p_str)
            p_str = re.sub(r"i\.e\.\s+", "ie ", p_str)

            # Removing URLs 
            # LangExtract tries to open these URLs when they occur in the document text
            # p_str= re.sub(r"http\S+", "", p_str) 

            p_str_chars = len(p_str)

            # If the paragraph does not end with final punctuation, accumulate it
            if not is_sentence_end(p_str):
                combined_paragraph = combine_paragraphs(combined_paragraph, p_str)
                combined_chars += p_str_chars
                continue

            # p_str ends with a sentence end; decide whether to process or accumulate it
            total_chars = combined_chars + p_str_chars
            if is_section_header(next_text):
                # Immediately process if the next text is a section header
                p_str = combine_paragraphs(combined_paragraph, p_str)
                combined_paragraph, combined_chars = "", 0
            elif total_chars < min_paragraph_size:
                # Not enough characters accumulated yet; decide based on next_text
                if next_text is None or (not is_page_text(next_text) and is_sentence_end(p_str)):
                    # End of document or next text item is not a text item and current paragraph ends with punctuation
                    # Process the paragraph and reset the accumulator even though this is a short paragraph
                    p_str = combine_paragraphs(combined_paragraph, p_str)
                    combined_paragraph, combined_chars = "", 0
                else:
                    # Combine with next paragraph
                    combined_paragraph = combine_paragraphs(combined_paragraph, p_str)
                    combined_chars = total_chars
                    continue
            else:
                # Sufficient characters: process the paragraph and reset the accumulator
                p_str = combine_paragraphs(combined_paragraph, p_str)
                combined_paragraph, combined_chars = "", 0

            p_str = combine_hyphenated_words(p_str)
            if p_str:  # Only add non-empty content
                para_num += 1
                add_paragraph(
                    p_str, 
                    #para_num, section_name, page_no, 
                    temp_docs, 
                    # temp_meta
                )
                page_no = None
            
            print("\n-__ Paragraph #", para_num, "; Section:", section_name)

            # print(p_str)
            # print(temp_docs, temp_meta)

### TODO translator class
            ## Translation
            if src_language_doc != "en":

                supported_languages = ["fr", "de", "es", "it", "itc", "nl"]
                if src_language_doc not in supported_languages:
                    print(f"Unsupported source language: {src_language_doc}. Continue with extraction on original text")
                    continue 

                print(f"Translating {src_language_doc} --> en")
                try:
                    p_str = tm.translate_2_english(src_language_doc, p_str)     
                except Exception as e:
                    print(f"! Cannot translate text, going to next chunk: {p_str}")
                    continue

            # collect cleaned text + meta data per doc            
            texts_clean.append(p_str)
            section_names.append(section_name)


        print("TEST Saving cleaned text as DoclingDocument to store text (p_str)+ meta (section title, page_no): ")
        # Initialize new DoclingDoc
        doclingdoc = DoclingDocument(schema_name="DoclingDocument",  version="1.0.0",name="My Custom Document")
        
        # write text and structural elements (titles, sections, and paragraphs) to doc     
        title_node = doclingdoc.add_title(text=filename_stem)
        for text, section_name in zip(texts_clean, section_names):
            print(section_name, ":", text)
            section_node = doclingdoc.add_heading(text=section_name, level=1, parent=title_node)
            doclingdoc.add_text(label=DocItemLabel.TEXT, text=text, parent=section_node)

        # Save doclingDocument as MD
        doclingdoc.save_as_markdown(cleaned_md_filepath)     

    print(f"Document cleaning took, {np.round((time.time() - time_cleaning) / 60, 1)} minutes")


    print("Chunking document...")
    chunk_iter = chunker.chunk(dl_doc=doclingdoc)  # NOTE cannot use chunker when re-created Docl.Document with cleaned text and old DoclingObject (from converter)
    doc = list(chunk_iter)
    # print(len(chunks), "chunks created with chunker.chunk(dl_doc=doclingdoc)")
    # print(len(texts_clean), "cleaned text items in doclingdoc.texts")
    
    # TODO 
    # test if contextualization improves model performance 

    # # apply contextualization (add section_name etc to chunk textfor better LLM unterstanding)
    # for i in range(len(chunks)):
    #     print(f"\nChunk {i} content before contextualization:\n{chunks[i].text}")
    
    #     ser_text = chunker.contextualize(chunk=chunks[i])
    #     ser_tokens = tokenizer.count_tokens(ser_text)
    #     print(f"chunker.contextualize(chunk) ({ser_tokens} tokens):\n{ser_text!r}")
    #     print()
    

    
    for chunk_no, chunk in enumerate(doc):

          

        # init dfs for interim results for each chunk
        ## TODO make as pydantic class with fixed attributes
        df_ci_geo_chunk = pd.DataFrame(
            columns=[
                "citation_id",
                "chunk_text",
                "ci_entity",
                "ci_entity_label",
                "geo_entity",
                "geo_entity_label",
                "token_distance",
            ]
        )    
        df_geollama_response = pd.DataFrame()

    
        print(f"\nProcessing chunk no. {chunk_no+1} / {len(doc)} of document: {filepath.name}")
        print("Chunk text: ", chunk.text)


        ## add punctuations back for e.g. and i.e. (for better reading in chunks by LLm-WF)
        chunk.text = re.sub(r"eg\s+", "e.g. ", chunk.text)
        chunk.text = re.sub(r"ie\s+", "i.e. ", chunk.text)
        

        print("Getting geolocations of CI assets ")       
        ## get most likely geolocation for each CI entity based on distance between tokens
        nlp_chunk = nlp(chunk.text)
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

                            # TODO calc distances between CI_TYPE ~ GEO entities based on word numbers and not entities (ie tokens)
                            if all_ents[ent_idx].label_ in ["GPE", "LOC"]:

                                ## check that GPE,LOC are not countries (too coarse info CI-GEO pair)
                                ## of GPE/LOC is country -> proceed with next GPE/LOC 
                                if all_ents[ent_idx].text in ci_geo_countries:
                                    continue

                                geo_idx = ent_idx
                                dist_ent_pair = np.abs(ci_idx - geo_idx)
                                distance_list.append(dist_ent_pair)
                                idx_in_chunk.append((ent_idx))
                                closest_pair_idx = np.argmin(distance_list)  # idx of closest GEO entity
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
                            "citation_id": citation,
                            "chunk_text": chunk.text,
                            "ci_entity": all_ents[ci_idx].text,
                            "ci_entity_label": all_ents[ci_idx].label_,
                            "geo_entity": all_ents[idx_in_chunk[closest_pair_idx]].text,
                            "geo_entity_label": all_ents[idx_in_chunk[closest_pair_idx]].label_,
                            "token_distance": distance_closest_pair,
                        }
                        df_ci_geo_chunk = pd.concat(
                            [  df_ci_geo_chunk, pd.DataFrame([result_dict])], ignore_index=True
                        )

                    except (IndexError, NameError) as e:
                        continue
        else:
            print("No CI_TYPE entities found in this chunk. Going to next chunk")
            continue

        
        ## post-process of DF CI-GEO pairs for each chunk
        unique_ci_geo_pairs = df_ci_geo_chunk.drop_duplicates(
            subset=["citation_id", "ci_entity", "geo_entity","chunk_text"])
        print("number of duplicates to remove:", len(df_ci_geo_chunk) - len(unique_ci_geo_pairs))

        df_ci_geo_chunk = df_ci_geo_chunk.drop_duplicates(
            subset=["citation_id", "ci_entity", "geo_entity", "chunk_text"]
            )# .reset_index(drop=True, inplace=True)



        print("\nSTEP 1")

        ## apply decoder on each chunk in document
        ## TODO replace iteration by loading entire document and use recursive chunking from langchain
        time2 = time.time()


        # clean up before applying CUDA
        gc.collect()
        torch.cuda.empty_cache() 
        torch.no_grad()
        # print(torch.cuda.memory_reserved() / 1e9)
    
        print(f"Starting geoparsing")

        ## Start geoparsing with geollama to verify Location        
        
        # extract locations
        # for d in tqdm(chunk.text):
        resp = geo_llama.geoparse(chunk.text)
        # TODO add here geollama prompt as var
        # TODO check if useful in model.py to set : model.use_checkpointing = True or  model.gradient_checkpointing_enable()
        
        # save results
        df_geollama_resp = pd.DataFrame(resp[0:])
        df_geollama_resp["citation_id"] = citation
        df_geollama_resp["chunk_id"] = chunk_no
        df_geollama_resp["chunk_text"] = chunk.text
        
        # maybe empty response
        try:
            print("Removing duplicated locations and countries from geollama response to keep only more specific location info")

            df_geollama_resp["name"] = list(set(df_geollama_resp["name"]))  # rm dublicates
            df_geollama_resp = df_geollama_resp[~df_geollama_resp["name"].isin(ci_geo_countries)]  # rm countries
            print("Toponyms from geollama\n", df_geollama_resp["name"])
        except:
            pass

        # saving all geollama repsonses , even when they cannot processed furthernfor later analysis
        df_geollama_response = pd.concat([df_geollama_response, df_geollama_resp], ignore_index=True)
        

        try:
            locations_per_chunk = df_geollama_resp["name"].to_list()
            lats_per_chunk = df_geollama_resp["latitude"].to_list()
            lons_per_chunk = df_geollama_resp["longitude"].to_list()
            RAGestimated_per_chunk = df_geollama_resp["RAG_estimated"].to_list()
        except:
            print("No locations extracted by geollama for this chunk. Going to next chunk\n", resp[0:])
            continue
        
        # sanity check that only cases with location infos are considered for further processing      
        if len(locations_per_chunk) == 0:
            print("geollama did not find any potential locations found for this chunk. Continue with next chunk")
            continue

        ## Input for LLM 1  -1st round
        if df_ci_geo_chunk.empty:
            context = [
                {
                    "text": chunk.text,
                    # "citation": citation,
                    # "title": filename_stem,
                    "entity_recognition": None,
                },
            ]

        else:  # TODO dissolve if else clause by making it in ci_locations: if df_ci_geo.chunk=j, xx, else None
            context = [
                {
                    "text": chunk.text,
                    # "citation": citation,
                    # "title": filename_stem,
                    "entity_recognition": df_ci_geo_chunk,
                },
            ]


        # print(os.environ["CUDA_VISIBLE_DEVICES"])
        # os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
        # print(os.environ["CUDA_VISIBLE_DEVICES"])

        # load LLM_1 prompt template
        static_prompt = em.load_prompt_template(template_filename="short_static_llama3_NER.txt")
        dynamic_prompt = em.load_prompt_template(template_filename="short_dynamic_llama3_NER.txt")
        # template_1 = em.load_prompt_template(template_filename="ci_loc_direct_impacts_geollama_step_1.txt")

        # apply LLM 1
        response = decoder_model_1.generate_response(
            question=question_1, context=context, # chunk_id=j
            static_prompt = static_prompt,
            dynamic_prompt=dynamic_prompt,
            max_new_tokens = 2048
        )
        # print(type(response ))
        # print(response)
        
        ## postprocess response
        try:
            try:
                df_resp = pp.postprocess_response(response[0])
            except Exception as e:
            # except (IndexError, ValueError) as e:
                df_resp = pp.postprocess_response(response)
            
            # save LLM response for each chunk  in dataframe for each document
            df_resp["citation_id"] = citation 
            df_resp["chunk_id"] = chunk_no  # add chunk id as identifier
            df_resp["ci_entity"] = df_ci_geo_chunk["ci_entity"] 
            df_resp["geo_entity"] = df_ci_geo_chunk["geo_entity"]
            df_resp["coord_potential_locations"] = str(dict(zip(locations_per_chunk, zip(lats_per_chunk, lons_per_chunk, RAGestimated_per_chunk)))) # INTERIM for verification of lat, lon 
            df_resp["chunk_text"] =  context[0]["text"]  # add (translated) chunk text for tracing back LLM response

            if not len(df_resp):
                print("LLM response is empty. Continue with next chunk.\n Response was:", response)
                continue

            # store reponse for chunk to interim df
            df_resp_step1 = pd.concat([df_resp_step1, df_resp], ignore_index=True)


        except (IndexError, ValueError) as e:
            print(f"Cannot add response: {e},\nFaulty response (before postprocessing):", response)
            # faulty response: e.g.  .., "location": "V" on satellite and online on radar"}, { ...}, {}
            responses_error_list.append({
                "citation_id": citation,
                "chunk_id": chunk_no,
                "response": response,
                "error": str(e)
            })
            print("Continue with next chunk\n")
            continue


        ## group Ci types into subgroups, 
        # TODO make nicer when df is empty
        if df_resp.isna().sum().sum() == 0:
            print("No infrastructure types extracted for this chunk, skip grouping into subgroups. Go to next chunk")
            continue

        if "infrastructure_group" in df_resp.columns:
            df_resp = pp.group_ci_types(df_resp, "infrastructure_type", "infrastructure_group", ci_patterns)
        else:
            df_resp["infrastructure_group"] = None
            df_resp = pp.group_ci_types(df_resp, "infrastructure_type", "infrastructure_group", ci_patterns)
        
        ## store cases which could not be grouped
        df_ci_cases_not_grouped = pd.concat([df_ci_cases_not_grouped, df_resp[df_resp['infrastructure_group'].isna()]], ignore_index=True)
        ## keep only records which are actually about CI (e.g., not theatre, stadion ..)
        df_resp.dropna(subset=["infrastructure_group"], inplace=True)
    

        # clean up after each chunk
        gc.collect()
        torch.cuda.empty_cache()  # mainly needed after training, small effect when LLM applied only for inference
        torch.no_grad()

        print(f"### ---- Processing time for chunk {chunk_no+1} STEP 1: {np.round((time.time() - time2) / 60, 1)} minutes ---- ###")

        print("\nSTEP 2")
        print("KEEPING location info from step 1 for further improvement")
        df_locs_org = df_resp.copy() 


        # clean up before applying CUDA
        gc.collect()
        torch.cuda.empty_cache() 
        print(torch.cuda.memory_reserved() / 1e9)
        torch.no_grad()

        # print(os.environ["CUDA_VISIBLE_DEVICES"])
        # os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
        # print(os.environ["CUDA_VISIBLE_DEVICES"])


        ## Input for LLM - 2nd round (geollama)
    
        # print(f"\nCHECKING input.text, df_resp, geollama resp \n{chunk.text}\n{df_resp},\n{locations_per_chunk}")
        static_prompt_geollama = em.load_prompt_template(template_filename="short_static_llama3_NER_geollm_step2.txt")
        dynamic_prompt_geollama = em.load_prompt_template(template_filename="short_dynamic_llama3_NER_geollm_step2.txt")
        
        context = [
            {
                "text": chunk.text,
                "citation": citation,
                "title": filename_stem, 
                "previous_response": df_resp,
                "potential_locations": locations_per_chunk,  # list of 1 or multiple location strings (no countries)      
                # "coordinates_potential_locations": list(zip(locations_per_chunk, lats_per_chunk, lons_per_chunk, RAGestimated_per_chunk)),
            },
        ]
        
        ## LLM 1 - step2
        response = decoder_model_2.generate_response(
            question=question_2, context=context, # chunk_id=j
            static_prompt = static_prompt_geollama,
            dynamic_prompt = dynamic_prompt_geollama,
            max_new_tokens = 2048
        )

        ## postprocess response
        try:
            try:
                df_resp_2 = pp.postprocess_response(response[0])
            except Exception as e:
            # except (IndexError, ValueError) as e:
                df_resp_2 = pp.postprocess_response(response)

            # save LLM response for each chunk  in dataframe for each document
            df_resp_2["citation_id"] = citation
            df_resp_2["chunk_id"] = chunk_no  # add chunk id as identifier
            df_resp_2["ci_entity"] = df_ci_geo_chunk["ci_entity"] 
            df_resp_2["geo_entity"] = df_ci_geo_chunk["geo_entity"]
            df_resp_2["coord_potential_locations"] = str(dict(zip(locations_per_chunk, zip(lats_per_chunk, lons_per_chunk, RAGestimated_per_chunk)))) # INTERIM for verification of lat, lon 
            df_resp_2["infrastructure_type_org"] = df_locs_org["infrastructure_type"] 
            df_resp_2["damage_org"] = df_locs_org["damage"] 
            df_resp_2["damage_value_org"] = df_locs_org["damage_value"] 
            df_resp_2["locations_org"] = df_locs_org["location"] 
            df_resp_2["chunk_text"] =  context[0]["text"]  # add (translated) chunk text for tracing back LLM response

            # collect resps for each doc
            print("CREATED final LLM response (STEP 1 & 2) successfully")
            df_resp_step2 = pd.concat([df_resp_step2, df_resp_2], ignore_index=True)
            # print(df_resp_2)

        except (IndexError, ValueError) as e:
            try: 
                print(f"Cannot add response: {e},\nFaulty response (before postprocessing):", response)
                # faulty response: e.g.  .., "location": "V" on satellite and online on radar"}, { ...}, {}
                responses_error_list.append({
                    "citation_id": citation,
                    "chunk_id": chunk_no,
                    "response": response, #.replace('\n', ''),
                    "error": str(e)
                })
            except:
                pass

        print(f"\n   Processing time for chunk {chunk_no} STEPs 1 & 2: {np.round((time.time() - time2) / 60, 1)} minutes\n")

        # clean up before applying CUDA
        gc.collect()
        torch.cuda.empty_cache() 
        print(torch.cuda.memory_reserved() / 1e9)
        torch.no_grad()

        # print(os.environ["CUDA_VISIBLE_DEVICES"])



    print(f"\n   Processing time for document {citation} STEP 1&2: {np.round((time.time() - time1) / 60, 1)} minutes\n")

    print(f"Safety: saving responses (Step 1 + 2) for doc: {citation} ")
    df_resp_step1.to_csv(f"./interim_results/llm1_geollm_step1_{citation}.csv", encoding='utf-8', index=False)
    df_resp_step2.to_csv(f"./interim_results/llm1_geollm_step2_{citation}.csv", encoding='utf-8', index=False)

    # clean up after each document
    gc.collect()
    torch.cuda.empty_cache()  # mainly needed after training, small effect when LLM applied only for inference
    torch.no_grad()

    # collecting all docs
    df_resp_step1_all = pd.concat([df_resp_step1_all, df_resp_step1], ignore_index=True)
    df_resp_step2_all = pd.concat([df_resp_step2_all, df_resp_step2], ignore_index=True)



print(f"\n\n ######## -------- CI impact extraction took {(time.time() - time0) / 60} minutes -------- ######## \n\n")


# %%


# %% [markdown]
# ### Finish run

# %%
print("Where CI types could not be grouped:\n", df_ci_cases_not_grouped)


# %%

df_resp_step2.info()

# %%
print(len(df_resp_step1))
unique_ci_geo_pairs = df_resp_step1.drop_duplicates()
print("number of duplicates to remove:", len(df_resp_step1) - len(unique_ci_geo_pairs))

df_resp_step1_nodupl = df_resp_step1.drop_duplicates( )# .reset_index(drop=True, inplace=True)
print(len(df_resp_step1_nodupl))
df_resp_step1_nodupl

# %%
print(len(df_resp_step2))
unique_ci_geo_pairs = df_resp_step2.drop_duplicates()
print("number of duplicates to remove:", len(df_resp_step2) - len(unique_ci_geo_pairs))

df_resp_step2_nodupl = df_resp_step2.drop_duplicates( )# .reset_index(drop=True, inplace=True)
print(len(df_resp_step2_nodupl))
df_resp_step2_nodupl


print(df_resp_step2.infrastructure_group.isna().sum())  # mostly cases which are not CI (theater, stadion..)
print(df_resp_step2.infrastructure_group.value_counts()) # four most common subgroups seems to be correct
# df_pred.infrastructure_group.unique()



gc.collect()
torch.cuda.empty_cache() 
torch.no_grad()
print(torch.cuda.memory_reserved() / 1e9)
# %%
print(torch.cuda.memory_reserved() / 1e9)


# %%
print("Chunk with erroneous responses:", responses_error_list.__len__())
df_responses_error = pd.DataFrame(responses_error_list)
# df_responses_error#.tail(3)


# %%
# df_responses_all_step2#.tail(3)

# %% [markdown]
# ### Saving

# %%
# PATH_LLM_DATA: Path = Path(s.PATH_DATA /"llm_outputs/")
# LLM_DATA_FILENAME: str = "llm_1_updprompt_distanceNER.csv"

# OUTPUT_LLM1_FILEPATH =  Path(PATH_LLM_DATA / LLM_DATA_FILENAME)#.replace(".csv", "_v2.csv"))
# OUTPUT_LLM1_FILEPATH 

# %%
safety_df = df_resp_step2_all.copy()

# save LLM 1 output to disk along with prompt text
if not os.path.isfile(OUTPUT_LLM1_FILEPATH):

    print(f"Saving prompt, LLM response and erroneous responses [.txt, .csv] to {OUTPUT_LLM1_FILEPATH} ...")

    with open(OUTPUT_LLM1_FILEPATH.parent / f"prompt_staticgeol_{OUTPUT_LLM1_FILEPATH.stem}.txt", "w") as f:
        f.write(static_prompt.render(context=context, question=question_1))
    try:
        with open(OUTPUT_LLM1_FILEPATH.parent / f"prompt_dynamicgeol_{OUTPUT_LLM1_FILEPATH.stem}.txt", "w") as f:
            f.write(dynamic_prompt.render(context=context, question=question_1))
    except Exception as e:
        print("UndefinedError: dynamic_prompt has probably no df_ci_geo (it is empty)")   
    with open(OUTPUT_LLM1_FILEPATH.parent / f"prompt_staticgeol_{OUTPUT_LLM1_FILEPATH.stem}.txt", "w") as f:
        f.write(static_prompt_geollama.render(context=context, question=question_2))
    with open(OUTPUT_LLM1_FILEPATH.parent / f"prompt_dynamicgeol_{OUTPUT_LLM1_FILEPATH.stem}.txt", "w") as f:
        f.write(dynamic_prompt_geollama.render(context=context, question=question_2))

    df_resp_step1_all.to_csv(f"{OUTPUT_LLM1_FILEPATH.stem}_step1.csv", encoding='utf-8', index=False)
    df_resp_step2_all.to_csv(f"{OUTPUT_LLM1_FILEPATH.stem}_step2.csv", encoding='utf-8', index=False)
    df_responses_error.to_csv(OUTPUT_LLM1_FILEPATH.parent / f"errors_{OUTPUT_LLM1_FILEPATH.stem}.csv", encoding='utf-8', index=False)

elif os.path.isfile(OUTPUT_LLM1_FILEPATH) and not os.path.isfile(OUTPUT_LLM1_FILEPATH.parent / f"{OUTPUT_LLM1_FILEPATH.stem}_v2.csv"):

    print(f"Output file {Path(OUTPUT_LLM1_FILEPATH).stem} already exists. Saving as {OUTPUT_LLM1_FILEPATH.stem}_v2 to avoid overwriting ...")

    # If the original files exists but the v2 file doesn't, create the v2 file
    with open(OUTPUT_LLM1_FILEPATH.parent / f"prompt_staticgeol_{OUTPUT_LLM1_FILEPATH.stem}_v2.txt", "w") as f:
        f.write(static_prompt.render(context=context, question=question_1))
    with open(OUTPUT_LLM1_FILEPATH.parent / f"prompt_dynamicgeol_{OUTPUT_LLM1_FILEPATH.stem}_v2.txt", "w") as f:
        f.write(dynamic_prompt.render(context=context, question=question_1))
    with open(OUTPUT_LLM1_FILEPATH.parent / f"prompt_staticgeol_{OUTPUT_LLM1_FILEPATH.stem}_v2.txt", "w") as f:
        f.write(static_prompt_geollama.render(context=context, question=question_2))
    with open(OUTPUT_LLM1_FILEPATH.parent / f"prompt_dynamicgeol_{OUTPUT_LLM1_FILEPATH.stem}_v2.txt", "w") as f:
        f.write(dynamic_prompt_geollama.render(context=context, question=question_2))
    
    df_resp_step1_all.to_csv(Path(OUTPUT_LLM1_FILEPATH.parent, f"{OUTPUT_LLM1_FILEPATH.stem}_step1_v2.csv"), encoding='utf-8', index=False)
    df_resp_step2_all.to_csv(Path(OUTPUT_LLM1_FILEPATH.parent, f"{OUTPUT_LLM1_FILEPATH.stem}_step2_v2.csv"), encoding='utf-8', index=False)
    df_responses_error.to_csv(Path(OUTPUT_LLM1_FILEPATH.parent / f"errors_{OUTPUT_LLM1_FILEPATH.stem}_v2.csv"), encoding='utf-8', index=False)

else:
    print(f"Output file {Path(OUTPUT_LLM1_FILEPATH).stem} already exists. Skip saving to avoid overwriting ...")




# %%
df_resp

# %%
df_resp

# %% [markdown]
# ## TODO 
# ## * does contextualization with chunker improves LLm performance?
# ## * check if LLm-extraction directly with doclingdoc (without chunker) [diff. txt-lenghts] is also possible
# 

# %%


# %%
resp.rpartition('"')[-3] + "]}"

# %%
# print(f"Safety: saving responses (Step 1 + 2) for doc: {citation} ")
# df_resp_step1.to_csv(f"./interim_results/llm1_geollm_step1_{citation}.csv", encoding='utf-8', index=False)
# df_resp_step2.to_csv(f"llm1_geollm_step2_{citation}.csv", encoding='utf-8', index=False)

print(df_resp_step2.chunk_text[0])



# %%


# %% [markdown]
# ## Doc. cleaning improve

# %%
# c = """ 
# blublub title\n\n\n

# HERE IS NEW SUBSECTION:\n

# (large-scale) societal dis- ruptions dis - ruptions (Garschagen and Sandholz, 2018; Hallegatte et al., 2019; Fekete and Sandholz, 2021), empirical evidence on the impacts of extreme weather events on these systems is still

# Published by Copernicus Publications on behalf of the European Geosciences Union.

# E. E. Koks et al.: Flood impacts to infrastructure

# limited. This brief communication provides an overview of the observed ﬂood impacts to large-scale infrastructure sys- tems during the 2021 mid-July western European ﬂood event and how reconstruction of these large-scale systems has pro- 




# HERE IS NEW SUBSECTION

# severely damaged railway line (between the vil- lages of Spa and Pepinster) was reopened again on 3 Octo- ber 2021 (Rozendaal, 2021b). In the Netherlands, no large- scale damage has been reported to transport infrastructure. A few national highways were partly ﬂooded (e.g. the A76 in both directions) or brieﬂy closed (&lt; 3 d) because of the po- tential of ﬂooding. Most likely due to relative low-ﬂow ve- locities, damage to Dutch national road infrastructure was limited. Several railway sections were closed (e.g. the rail-

# way section between Maastricht and Liége) and some dam- age occurred to the railway infrastructure, in particular to the electronic “track circuit” devices and saturated railway em- bankments (Prorail, 2021).

# """
# #c = c.replace(r"\n", r" ")   # Isssue replaces also multipelinebreas eg before subsection
# c = re.sub(r"([^\s-])\n([^\s-])", r"\1 \2", c) # replace linebreak symbols when they occur just once, with whitespace (two linebreaks - probably new subsection)
# c = c.replace("/\n{2,}/g", "\n")  # remove linebreaks only when they occurred just once, but not for multiple linebreaks (e.g. before subsection)
# # Matches \n not preceded or followed by \n
# # c = re.sub(r"(?<!\n)\n(?!\n)", r"\n", c)  # remove linebreaks only when the yoccured just once, but not for multiple linebreaks (e.g. before subsection)
# c = re.sub(r"\s+", " ", c)  # replace >1 whitespaces with single whitespace

# # c = c.replace(r"\w*- ", "\w*-", c)  # removes any word followed by "-"
# # c = re.sub(r"([^\s-])-\n([^\s-])", r"\1\2", c)  # remove hypen and linebreaks TODO test with koks sentences
# c = re.sub(r"([^\s-])- ([^\s-])", r"\1-\2", c)  # remove hypens in the middle of lines

# c 


# # 0090- some weird breaks-
# # And some long sentences 
# # which are not separated by dots but by line breaks and hyphens 



# %%
# !uv pip install "unstructured[pdf]"  # unstructured  #langchain-unstructured #langchain-community
# # # !uv add langchain
# !uv lock
# !uv sync
# from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
# #from langchain.loaders import , UnstructuredFileLoader

# loader = DirectoryLoader(str(Path(DOCS_DIR)),  loader_cls=UnstructuredFileLoader, show_progress=True)
# pdf_docs = loader.load()
# # glob=glob("*.pdf"),
# print(f"Number of Documents: {len(pdf_docs)}")

# # convert the different layouts of the pdf files into unified markdown format incl. sub/section titles, tables, caption text etc
# for idx, doc in enumerate(pdf_docs, start=0):
#     print(doc)

# %%
# pdf_filepath = Path("Lloyd's List 2024 - Port of Valencia reopens after devastating floods.pdf")
# # Path(DOCS_DIR, "Lloyd's List 2024 - Port of Valencia reopens after devastating floods.pdf")
# print(pdf_filepath)
# print("Remove reference section")

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
# pdf_text = converter.convert(pdf_filepath).document
    
# # loader = DoclingLoader("/beegfs/scratch/a-buch/_PROJECTS/data/text_sources/Lloyd's List 2024 - Port of Valencia reopens after devastating floods.pdf")  # use chunks from Docling.Loader
# # pdf_doc = loader.load()

# %%




#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Global variables and logger functions"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

from datetime import datetime
import os
import re
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import subprocess

# local or remote machine 
hostname = subprocess.run(['hostname'])

class Settings():
# class Settings(BaseSettings):
    # paths
    PATH_SRC: str = "./src"
    ## store logs and data outside of the repository
    PATH_LOGS: str = "../logs/"
    PATH_DATA: str = "../data/"
    PATH_EVALUATION: str = PATH_DATA + 'evaluation/'


    #  Define data dir where tags.csv and domain-expertise derived tag lists are found 
    PATH_VALID_DATA: Path = Path(PATH_EVALUATION + 'manual_extracted')
    VALID_DATA_FILENAME: str = 'table_ci_impacts_sm.csv'
    
    PATH_EVAL_RESULT: Path = Path(PATH_DATA + 'evaluation_results')
    os.makedirs(PATH_EVAL_RESULT, exist_ok=True)
    SIMILARITY_FILENAME: str = 'llm1_ci_impact_evaluation.parquet'
    SIMILARITY_LX_FILENAME: str = 'lx1_ci_impact_evaluation.parquet'
    # SIMILARITY_FILENAME: str = 'llama3_ci_impact_evaluation.parquet' # replace with lx_modelname_xxx

    PATH_LLM_DATA: Path = Path(PATH_DATA + "llm_outputs/")
    # LLM_DATA_FILENAME: str = 'llm1_43docs_eval.csv'
    LLM_DATA_FILENAME: str = f'llm1_{datetime.now().strftime("%Y-%m-%d")}.csv'
    
    PATH_LX_DATA: Path = Path(PATH_DATA + "langextract_output/")
    os.makedirs(PATH_LX_DATA, exist_ok=True)
    # LX_DATA_FILENAME: str = 'llama3_48_documents_2026-01-28.csv'   #  replace with lx_modelname_xxx
    LX_DATA_FILENAME: str = f"lx1_mix_cigeo_{datetime.now().strftime('%Y-%m-%d')}.csv"

    PATH_PROMPTS: Path = Path("./prompt_templates/")
    PROMPT_DIRECT_FILENAME: str = "ci_loc_direct_impacts.txt"
    NER_PATTERNS_FILEPATH: Path = Path("./" + "ner_patterns.jsonl")
    CI_GEO_PAIRS_FILENAME: str = "extracted_ci_geo_entities.csv"

    HUGGINGFACE_TOKEN: str
    model_config = SettingsConfigDict(env_file=".env")  # load HUGGINGFACE_TOKEN

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Use transformer model (roberta) for fast and precise NER recognition for english language [only with GPU]
    # performance: https://spacy.io/models/en#en_core_web_trf
    try:
        subprocess.check_output("nvidia-smi")
        SPACY_MODEL: str = "en_core_web_trf"
    except Exception:
        SPACY_MODEL: str = "en_core_web_lg"

    if hostname.stdout == "a-buch-ThinkPad-X1-Extreme-Gen-4i":
        print("Running on local machine")

        # set working dir
        os.chdir("/home/a-buch/Documents/TUB_DWN/_PROJECTS/CI-impacts-information-retrieval/")
    
        # HF directory
        HF_HOME_DIR: str = "/home/a-buch/Documents/TUB_DWN/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror/"

        # settings for CUDA and PYTORCH
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="0"   #  nvidia gpu
        os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
        # # %env TORCH_CUDA_ARCH_LIST=8.6

        # settings for distributed computing
        os.environ["WORLD_SIZE"]="1"
        os.environ["RANK"]="0"
        os.environ["LOCAL_RANK"]="0"
        # NOTE: # WORLD_SIZE: each GPU corresponds to one process (world = no. of processes within a group), processes communicate with each other enabling eg., distributed training
        # NOTE: # RANK: IDs of the processes, ranging from 0 up to WORLD_SIZE - 1

    # elif re.findall("node*|gpu*", hostname) ==TRUE:  # TODO adapt pattern search
    else:
        print("Running on TUB Cluster")
        HF_HOME_DIR: str = "/beegfs/home/users/a/a-buch/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror"
        HF_TOKEN_PATH: str = "/beegfs/home/users/a/a-buch/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror/token"

    # else:
    #     print("Could not identify if code is executed on local or remote machine, adapt hostname and paths accordingly!")


settings = Settings()
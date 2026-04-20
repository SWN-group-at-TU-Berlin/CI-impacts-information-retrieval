#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Global variables and logger functions"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

from datetime import datetime
import os
import re
from pathlib import Path
# from pydantic_settings import BaseSettings
import subprocess



# local or remote machine 
hostname = subprocess.run(['hostname'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()

class Settings():
#class Settings(BaseSettings):

    if hostname == "abuch-ThinkPad-X1-Extreme-Gen-4i":
        print("Running on local machine")
        PATH_DATA: Path = Path("../data/")
    else:
        print("Running on TUB cluster")
        PATH_DATA: Path = Path("/beegfs/scratch/a-buch/_PROJECTS/data/")
    
    PATH_SRC: Path = Path("./src")
    ## store logs and data outside of the repository
    PATH_LOGS: Path = Path("../logs/")
    
    PATH_PROMPTS: Path = Path("./prompt_templates/")
    PROMPT_DIRECT_FILENAME: str = "ci_loc_direct_impacts.txt"
    NER_PATTERNS_FILEPATH: Path = Path("./ner_patterns.jsonl")
    CI_GEO_PAIRS_FILENAME: str = "extracted_ci_geo_entities.csv"    
    

    PATH_LLM_DATA: Path = Path(PATH_DATA / "llm_outputs/")
    LLM_DATA_FILENAME: str = "llm_1_dNER_fixNER.csv" #"llm_1_half_validDS.csv"
    #LLM_DATA_FILENAME: str = f'llm1_{datetime.now().strftime("%Y-%m-%d")}.csv'
    
    PATH_LX_DATA: Path = Path(PATH_DATA / "langextract_output/")
    os.makedirs(PATH_LX_DATA, exist_ok=True)
    LX_DATA_FILENAME: str = f"lx1_mix_cigeo_{datetime.now().strftime('%Y-%m-%d')}.csv"
    # LX_DATA_FILENAME: str = 'llama3_48_documents_2026-01-28.csv'   #  replace with lx_modelname_xxx


    PATH_VALID_DATA: Path = Path(PATH_DATA / 'evaluation/manual_extracted')
    VALID_DATA_FILENAME: str = 'table_ci_impacts_sm.csv'
    PATH_EVAL_RESULT: Path = Path(PATH_DATA / 'evaluation_results')
    os.makedirs(PATH_EVAL_RESULT, exist_ok=True)

    # NOTE SIMILARITY_[LX]_FILENAME is set based on LLM_DATA_FILENAME or LX_DATA_FILENAME 
    SIMILARITY_LLM_FILENAME: str = f'smlrty_{LLM_DATA_FILENAME.replace(".csv", ".parquet")}'
    SIMILARITY_LX_FILENAME: str = f'smlrty_{LX_DATA_FILENAME.replace(".csv", ".parquet")}'
    # SIMILARITY_FILENAME: str = 'llama3_ci_impact_evaluation.parquet' # replace with lx_modelname_xxx


    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    BATCH_SIZE: int = 5  # max for my local GPU (as max. is 6GB)
    
        
    # settings for CUDA and PYTORCH
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"   #  nvidia gpu
    os.environ["PYTORCH_ALLOC_CONF"]="expandable_segments:True" ## improve memory allocation
    # # %env TORCH_CUDA_ARCH_LIST=8.6

    # settings for debugging CUDA errors (pinpoint exact line of error)
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 

    # # settings for distributed computing
    # os.environ["WORLD_SIZE"]="1"
    # os.environ["RANK"]="0"
    # os.environ["LOCAL_RANK"]="0"
    # NOTE: # WORLD_SIZE: each GPU corresponds to one process (world = no. of processes within a group), processes communicate with each other enabling eg., distributed training
    # NOTE: # RANK: IDs of the processes, ranging from 0 up to WORLD_SIZE - 1


    # Use transformer model (roberta) for fast and precise NER recognition for english language [only with GPU]
    # performance: https://spacy.io/models/en#en_core_web_trf
    try:
        subprocess.check_output("nvidia-smi")
        # FIXME workaround to not update torch which causes issues with flash-attn
        #"solution: en_core_web_trf should be used as it is faster and maybe more precise"
        SPACY_MODEL: str = "en_core_web_trf"
    except Exception:
        print("couldnt load large spacy model")
        SPACY_MODEL: str = "en_core_web_lg "
    
    if hostname == "abuch-ThinkPad-X1-Extreme-Gen-4i":

        # set working dir
        os.chdir("/home/a-buch/Documents/TUB_DWN/_PROJECTS/CI-impacts-information-retrieval/")
    
       
        # HF directory
        HF_HOME_DIR: str = "/home/a-buch/Documents/TUB_DWN/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror/hub"
        HF_TOKEN_PATH: str = "/home/a-buch/Documents/TUB_DWN/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror/token"

    # elif re.findall("node*|gpu*", hostname) ==TRUE:  # TODO adapt pattern search
    else:
        print("Running on TUB Cluster")
        HF_HOME_DIR: str = "/beegfs/home/users/a/a-buch/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror/hub"
        HF_TOKEN_PATH: str = "/beegfs/home/users/a/a-buch/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror/token"

        # DATA DIR
        PATH_DATA: Path = Path("/beegfs/scratch/a-buch/_PROJECTS/data/")
    # else:
    #     print("Could not identify if code is executed on local or remote machine, adapt hostname and paths accordingly!")


settings = Settings()
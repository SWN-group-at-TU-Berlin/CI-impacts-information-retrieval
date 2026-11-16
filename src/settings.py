#!/usr/bin/env python
"""Global variables and logger functions"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import logging
import os
import subprocess
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# local or remote machine
hostname = subprocess.run(["hostname"], stdout=subprocess.PIPE).stdout.decode("utf-8").strip()


class Settings(BaseSettings):
    # fields can be overwritten by environment variables or by the .env file in the project root (e.g. LLM1_BASE_URL)
    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parents[1] / ".env", extra="ignore")

    if hostname == "abuch-ThinkPad-X1-Extreme-Gen-4i":
        print("Running on local machine")
        PATH_DATA: Path = Path("../data/")
    else:
        print("Running on TUB cluster")
        PATH_DATA: Path = Path("/beegfs/scratch/a-buch/_PROJECTS/data/")

    PATH_LOGS: Path = Path("../logs/")
    DOCLING_ARTIFACTS: Path = Path("../docling_artifacts")

    NER_PATTERNS_FILEPATH: Path = Path("./ner_patterns.jsonl")
    CI_GEO_PAIRS_FILENAME: str = "extracted_ci_geo_entities.csv"

    PATH_LLM_DATA: Path = Path(PATH_DATA / "llm_outputs/")
    LLM_DATA_FILENAME: str = "llm_1_dNER_fixNER.csv"  # "llm_1_half_validDS.csv"

    PATH_VALID_DATA: Path = Path(PATH_DATA / "evaluation/manual_extracted")
    VALID_DATA_FILENAME: str = "table_ci_impacts_sm.csv"
    PATH_EVAL_RESULT: Path = Path(PATH_DATA / "evaluation_results")

    SIMILARITY_LLM_FILENAME: str = f"smlrty_{LLM_DATA_FILENAME.replace('.csv', '.parquet')}"

    # self-hosted LLMs served by vLLM (OpenAI-compatible API), see scripts/serve_vllm_*.sbatch
    # NOTE: from the local machine reach the HPC servers via ssh tunnel, e.g. ssh -N -L 8000:<node>:8000 -L 8001:<node>:8001 <hpc-login>
    LLM1_BASE_URL: str = os.getenv("LLM1_BASE_URL", "http://localhost:8000/v1")  # use env var if set otherwise default
    JUDGE_BASE_URL: str = os.getenv("JUDGE_BASE_URL", "http://localhost:8001/v1")
    LLM1_MODEL: str = "openai/gpt-oss-20b"  # served model name, see `vllm serve --served-model-name`
    JUDGE_MODEL: str = "google/gemma-4-31B-it"  # served model name, see `vllm serve --served-model-name`
    VLLM_API_KEY: str | None = None  # set in .env or as env variable, same key as `vllm serve --api-key`

    CHUNK_OVERLAP: int = 50
    BATCH_SIZE: int = 5  # max for my local GPU (as max. is 6GB)

    # settings for CUDA and PYTORCH
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #  nvidia gpu
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  ## improve memory allocation

    # settings for debugging CUDA errors (pinpoint exact line of error)
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Use transformer model (roberta) for fast and precise NER recognition for english language [only with GPU]
    # performance: https://spacy.io/models/en#en_core_web_trf
    try:
        subprocess.check_output("nvidia-smi")
        # FIXME workaround to not update torch which causes issues with flash-attn
        # "solution: en_core_web_trf should be used as it is faster and maybe more precise"
        SPACY_MODEL: str = "en_core_web_trf"
    except Exception:
        print("couldnt load large spacy model")
        SPACY_MODEL: str = "en_core_web_lg "

    # HF directory
    if hostname == "abuch-ThinkPad-X1-Extreme-Gen-4i":
        print("Running on local machine")

        # set working dir
        os.chdir("/home/a-buch/Documents/TUB_DWN/_PROJECTS/CI-impacts-information-retrieval/")

        # HF directory
        HF_HOME_DIR: str = "/home/a-buch/Documents/TUB_DWN/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror/hub"
        HF_TOKEN_PATH: str = "/home/a-buch/Documents/TUB_DWN/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror/token"
    else:
        print("Running on TUB Cluster")
        HF_HOME_DIR: str = "/beegfs/home/users/a/a-buch/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror"
        HF_TOKEN_PATH: str = "/beegfs/home/users/a/a-buch/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror/token"


settings = Settings()


def init_logger(name):
    """
    Set up a logger instance
    Modified version based on code from <christina.ludwig@uni-heidelberg.de> for SM2T project
    name (str): Name of logger
    log_file (str): path to log file
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m-%d-%Y %I:%M:%S",
    )
    # avoid duplicated handlers (and log lines) when the logger is initialized multiple times, e.g. module reloads
    if logger.handlers:
        return logger

    # Add stream handler
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.INFO)
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)

    # Add file handler
    os.makedirs(settings.PATH_LOGS, exist_ok=True)
    log_file = f"{settings.PATH_LOGS}/logs_{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

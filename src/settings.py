#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Global variables and logger functions"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

from pydantic_settings import BaseSettings, SettingsConfigDict
import subprocess


class Settings(BaseSettings):
    # paths
    PATH_SRC: str = "./src"
    ## store logs and data outside of the repository
    PATH_LOGS: str = "../logs/"
    PATH_DATA: str = "../data/"

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


settings = Settings()

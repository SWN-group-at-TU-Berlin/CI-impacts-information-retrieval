#!/usr/bin/env python
"""Settings of the project and logger function.

The settings are read (highest priority first) from
1. environment variables,
2. the .env file in the project root (secrets, e.g. VLLM_API_KEY),
3. the config file configs/config.yaml (paths, LLM servers), or the file given by the env variable CI_IMPACTS_CONFIG.
"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import logging
import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = Path(os.getenv("CI_IMPACTS_CONFIG", PROJECT_ROOT / "configs" / "config.yaml"))

# settings for CUDA and PyTorch, can be overwritten by environment variables
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")  # improve memory allocation
# debugging of CUDA errors (pinpoint exact line of error)
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")


class Settings(BaseSettings):
    """Defaults of the settings, overwritten by configs/config.yaml, .env and environment variables (see module docstring)"""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", yaml_file=CONFIG_FILE, extra="ignore")

    # data, results and logs
    PATH_DATA: Path = Path("../data")
    PATH_RESULTS: Path = Path("../results")
    PATH_LOGS: Path = Path("../logs")

    # models stored on disk
    DOCLING_ARTIFACTS: Path = Path("../docling_artifacts")
    SPACY_MODEL_PIPELINE: Path = Path("spacy_model_pipeline")  # TODO move outside project folder, see related git issue
    HF_CACHE_DIR: Path | None = None  # None: default cache of Hugging Face

    # self-hosted LLMs served by vLLM (OpenAI-compatible API), see run_scripts/serve_vllm_*.sbatch
    LLM1_BASE_URL: str = "http://localhost:8000/v1"
    JUDGE_BASE_URL: str = "http://localhost:8001/v1"
    LLM1_MODEL: str = "openai/gpt-oss-20b"  # served model name, see `vllm serve --served-model-name`
    JUDGE_MODEL: str = "google/gemma-4-31B-it"
    VLLM_API_KEY: str | None = None  # secret: set in .env, same key as `vllm serve --api-key`

    @field_validator("PATH_DATA", "PATH_RESULTS", "PATH_LOGS", "DOCLING_ARTIFACTS", "SPACY_MODEL_PIPELINE", "HF_CACHE_DIR")
    @classmethod
    def resolve_path(cls, path: Path | None) -> Path | None:
        """Relative paths are resolved from the project root, so that the settings do not depend on the working directory"""
        if path is None:
            return None
        path = Path(path).expanduser()
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # priority: arguments > environment variables > .env > config.yaml
        return init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls), file_secret_settings


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

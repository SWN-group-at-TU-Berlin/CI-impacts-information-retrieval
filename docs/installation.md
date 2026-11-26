# Installation and usage – step by step

1. [Install the project](#1-install-the-project)
2. [Set your paths, servers and secrets](#2-set-your-paths-servers-and-secrets)
3. [Docling models](#3-docling-models-docling_artifacts)
4. [spaCy model pipeline](#4-spacy-model-pipeline-spacy_model_pipeline)
5. [Start the LLM servers (vLLM) and create the API key](#5-start-the-llm-servers-vllm-and-create-the-api-key)
6. [Add your documents](#6-add-your-documents)
7. [Run the pipeline](#7-run-the-pipeline)

All settings mentioned below are listed in [configuration.md](configuration.md). If something does not work, see [troubleshooting.md](troubleshooting.md).

## 1. Install the project
```bash
git clone https://github.com/SWN-group-at-TU-Berlin/CI-impacts-information-retrieval.git
cd CI-impacts-information-retrieval

# create venv and install dependencies, activate venv
uv sync
source .venv/bin/activate
```
`uv sync` creates the virtual environment `.venv` with all packages of `pyproject.toml` (incl. the development tools, use `uv sync --no-dev` without them).
Run all commands with `uv run ...`.

## 2. Set your paths, servers and secrets
All user settings are stored in [`configs/config.yaml`](../configs/config.yaml). Adapt the paths (`PATH_DATA`, `PATH_RESULTS`, `PATH_LOGS`, ...) and the URLs of the LLM servers to your machine, see the [table of all entries](configuration.md#configsconfigyaml).

Secrets do not belong into the config file. Copy the template and fill in your values (at least `VLLM_API_KEY`, see step 5):
```bash
cp .env_example .env      # .env is ignored by git
```
See [configuration.md](configuration.md#env-secrets) for all `.env` entries.

## 3. Docling models (`DOCLING_ARTIFACTS`)
The PDF processing (Docling with OCR) loads its models from the folder `DOCLING_ARTIFACTS` and does **not** download missing models by itself. Download the layout model and the EasyOCR models once (see [Docling: model prefetching](https://docling-project.github.io/docling/usage/advanced_options/#model-prefetching-and-offline-usage)):
```bash
source .venv/bin/activate # activate environment if you havent done already
uv run docling-tools models download layout easyocr -o ../docling_artifacts   # use the path of DOCLING_ARTIFACTS
```
The folder then contains the layout model (`docling-project--docling-layout-heron`) and the folder `EasyOcr/` with the text detection model (`craft_mlt_25k.pth`) and the recognition models for English and Latin-script languages (`english_g2.pth`, `latin_g2.pth`).

## 4. spaCy model pipeline (`SPACY_MODEL_PIPELINE`)
The NER pre-filter uses the spaCy pipeline [`en_core_web_trf`](https://spacy.io/models/en#en_core_web_trf) (English, transformer-based) extended by the components `merge_entities` and `merge_noun_chunks`. The CI patterns (`ner_patterns.jsonl`) are added when the pipeline is loaded.

The pipeline is **created automatically** on the first run if the folder `SPACY_MODEL_PIPELINE` does not exist (download of about 460 MB). To create it manually, run
```bash
source .venv/bin/activate # activate environment if not done already
uv run python -m spacy download en_core_web_trf
uv run python -c "
import spacy
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('merge_entities')
nlp.add_pipe('merge_noun_chunks')
nlp.to_disk('spacy_model_pipeline')   # path of SPACY_MODEL_PIPELINE
"
```

### Translation models (no action needed)
Documents that are not in English are translated to English with the [Helsinki-NLP opus-mt](https://huggingface.co/Helsinki-NLP) models (`Helsinki-NLP/opus-mt-<language>-en`). The language is detected automatically.

Currently supported source languages:

| Language | Code | Model |
|---|---|---|
| French | `fr` | `Helsinki-NLP/opus-mt-fr-en` |
| German | `de` | `Helsinki-NLP/opus-mt-de-en` |
| Spanish | `es` | `Helsinki-NLP/opus-mt-es-en` |
| Italian | `it` | `Helsinki-NLP/opus-mt-it-en` |
| Dutch | `nl` | `Helsinki-NLP/opus-mt-nl-en` |

Text in other languages is not translated and passed to the LLMs in its original language. The models are downloaded automatically on first use into `HF_CACHE_DIR`.
OCR of scanned PDFs works for the same languages (EasyOCR with `fr`, `de`, `es`, `it`, `nl`).

## 5. Start the LLM servers (vLLM) and create the API key
The extraction LLM and the judge run as two OpenAI-compatible [vLLM servers](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html), protected by an API key.

1. **Set an API key**: vLLM does not issue keys, the key is a secret string that you choose yourself and that is passed to the server with `vllm serve --api-key <key>` (see the [vLLM docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html)). Choose a long key that is hard to guess and enter it in `.env`:
   ```bash
   VLLM_API_KEY=<your_key>
   ```
   The scripts in `run_scripts/` start the servers with this key and the pipeline sends the same key to the servers. Do not commit the key (`.env` is ignored by git).

2. **Install vLLM**
in a separate environment on the GPU machine (vLLM pins its own torch/CUDA versions), see the [vLLM installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/).
Gemma 4 needs vLLM >= 0.19.1 (thus the requirement for CUDA>=12.8) .

   On an HPC cluster, install vLLM with a batch job, but you can also use an interactive `srun` session. Below is an example for the vllm installation by using a batch job:
   ```bash
   # from the project root; logs go to ../logs/ (create it once: mkdir -p ../logs)
   sbatch run_scripts/install_vllm.sbatch
   ```
   The job does the following:
   - creates a Python 3.12 venv with `uv` in `VLLM_ENV` (default `$HOME/_PROJECTS/vllm/.venv`, which is the default the serve scripts expect)
   - installs `vllm==$VLLM_VERSION` (default `0.19.1`) with `--torch-backend=auto`, so `uv` picks the torch CUDA build that matches the node's NVIDIA driver
   - checks that `torch` sees the GPU and that `vllm` can be imported (output in `../logs/install-vllm-<jobid>.out`)

   To install into another folder or a different version, set the variables when you submit the job (see [run-script variables](configuration.md#run-script-variables)):
   ```bash
   VLLM_ENV=/work/<user>/vllm/.venv UV_CACHE_DIR=/work/<user>/.cache/uv sbatch run_scripts/install_vllm.sbatch
   VLLM_VERSION=<version> REINSTALL=1 sbatch run_scripts/install_vllm.sbatch   # REINSTALL=1 recreates the venv
   ```
   If you change `VLLM_ENV`, pass the same path to the serve scripts later (`VLLM_ENV=<path> sbatch run_scripts/serve_vllm_...`).
   Notes:
   - Adapt the `#SBATCH` header (partition, account, GPU type) to your cluster. Install on the same GPU type you will use for serving.
   - `uv` must be installed in your home directory (`~/.local/bin`).
   - The compute node needs internet access (PyPI, download.pytorch.org). If it has none, run the `uv venv ...` and `uv pip install ...` commands from the script on the login node instead.

3. **Start the servers**.
On a HPC cluster use one of the scripts in `run_scripts/` as starting point and adapt it to your needs (e.g., change the GPU type).
The bash scripts start both servers and afterwards the CI-extraction pipeline (`examples/extract_ci_impacts.py`) in the same job:
   ```bash
   ## Example run scripts
   sbatch run_scripts/serve_vllm_both_h200.sbatch      # 1x H200 - 141 GB
   sbatch run_scripts/serve_vllm_both_2xa100.sbatch    # 2x A100 - each 40 GB
   ```
   The scripts read `VLLM_API_KEY` from `.env`.
   Paths can be set as variables in the scripts, e.g. `PROJECTROOT=<project folder> VLLM_ENV=<vLLM venv>`.
   Adapt the SLURM settings in the `#SBATCH` header of the scripts to your cluster.
   The pipeline runs on the same GPU node as the servers and reaches them via `localhost` (default URLs in `config.yaml`).

4. **Check if the pipeline could work:**
before a full run, test at least one vllm server and if the respective LLM is responding: `sbatch run_scripts/smoke_test_gptoss.sbatch`.
The command starts the gpt-oss server and runs `run_scripts/smoke_test_gptoss.py` against it.

## 6. Add your documents
Store the documents in `<PATH_DATA>/text_sources/`:
- **PDFs**: named `<Authors> <Year> - <Title>.pdf`, e.g. `Koks 2022 - Brief communication.pdf`. The citation of the extracted records (e.g. `Koks 2022`) is taken from the file name.
- **Web pages** (optional): a table `text_sources_html.csv` with the columns `Authors`, `Year`, `Title`, `URL`. The pages are downloaded and cleaned by the pipeline.

The cleaned documents are automatically stored in `<PATH_DATA>/parsed_documents/` and reused in later runs.

## 7. Run the pipeline
The pipeline runs on the GPU node: the scripts of step 5 start both vLLM servers and then run [`examples/extract_ci_impacts.py`](../examples/extract_ci_impacts.py) in the same job:
```bash
# Example:
sbatch run_scripts/serve_vllm_both_h200.sbatch
```
To run another script, pass it with `SCRIPT=<script> sbatch ...` (however currently a change of the script is less needed as only one example for the CI-extraction pipeline exists: [`examples/extract_ci_impacts.py`](../examples/extract_ci_impacts.py) ).
In the settings at the top of the example you can restrict the run to a few documents (`SAMPLE_DOCUMENTS`) and switch on the processing of web pages (`PROCESS_HTML`), see [example-script settings](configuration.md#example-script-settings).
<!-- The notebook [`examples/extract_ci_impacts.ipynb`](../examples/extract_ci_impacts.ipynb) shows the same steps one by one; run it on the GPU node while the servers are running. -->

Results in `PATH_RESULTS`:
- `llm_chain_judge_<citation>.csv` – one row per extracted infrastructure case (type, amount, name, location, damage, cascading impacts) with the verdict of the judge. All columns and their allowed values are described in [extracted_features.md](extracted_features.md).
- `llm_responses_errors.csv` – chunks for which the LLM call failed

Logs are written to `PATH_LOGS`.

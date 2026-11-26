# Configuration

Reference of all settings. For the order in which to set them up, see [installation.md](installation.md).

- [`configs/config.yaml`](#configsconfigyaml) – paths, LLM servers and models
- [`.env` (secrets)](#env-secrets) – API key and Hugging Face settings
- [Run-script variables](#run-script-variables) – variables of the Slurm scripts in `run_scripts/`
- [Example-script settings](#example-script-settings) – settings at the top of `examples/extract_ci_impacts.py`

## `configs/config.yaml`
All user settings are stored in [`configs/config.yaml`](../configs/config.yaml) and read by `src/settings.py`. Adapt the entries to your machine:

| Entry | Default | Meaning |
|---|---|---|
| `PATH_DATA` | `../data` | input documents and cleaned documents (see [step 6](installation.md#6-add-your-documents)) |
| `PATH_RESULTS` | `../results` | extracted records of the pipeline |
| `PATH_LOGS` | `../logs` | log files |
| `DOCLING_ARTIFACTS` | `../docling_artifacts` | Docling models for the PDF processing (see [step 3](installation.md#3-docling-models-docling_artifacts)) |
| `SPACY_MODEL_PIPELINE` | `spacy_model_pipeline` | spaCy pipeline of the NER pre-filter (see [step 4](installation.md#4-spacy-model-pipeline-spacy_model_pipeline)) |
| `HF_CACHE_DIR` | `null` | cache of the translation models (see [Translation models](installation.md#translation-models-no-action-needed)), `null` = Hugging Face default (`~/.cache/huggingface/hub`) |
| `LLM_EXTRACTOR_BASE_URL`, `JUDGE_BASE_URL` | `http://localhost:8000/v1`, `http://localhost:8001/v1` | URLs of the two vLLM servers (see [step 5](installation.md#5-start-the-llm-servers-vllm-and-create-the-api-key)) |
| `LLM_EXTRACTOR_MODEL`, `JUDGE_MODEL` | `openai/gpt-oss-20b`, `google/gemma-4-31B-it` | served model names |

- **Relative paths** are resolved from the project root (the folder with `pyproject.toml`), so `../data` is the folder `data` next to the project folder. Absolute paths (e.g. `/scratch/<user>/data`) and `~` work as well.
- **Overwriting single entries**: instead of editing `config.yaml`, single entries can also be set in the `.env` file, e.g. `PATH_DATA=/scratch/<user>/data`. A different config file can be used with `export CI_IMPACTS_CONFIG=configs/<your_config>.yaml`.
- **Priority**: environment variables > `.env` > `configs/config.yaml`.

## `.env` (secrets)
Secrets do not belong into the config file. Copy the template and fill in your values:
```bash
cp .env_example .env      # .env is ignored by git
```

| `.env` entry | Needed for |
|---|---|
| `VLLM_API_KEY` | access to the vLLM servers (see [step 5](installation.md#5-start-the-llm-servers-vllm-and-create-the-api-key)) |
| `HF_TOKEN` (optional) | download of gated models from Hugging Face, e.g. gemma for the judge on the GPU machine ([create a token](https://huggingface.co/docs/hub/security-tokens) and accept the license on the [model page](https://huggingface.co/google/gemma-4-31B-it)) |
| `HF_HOME` (optional) | cache folder of the LLM weights on the GPU machine, default `~/.cache/huggingface` |

## Run-script variables
The Slurm scripts in [`run_scripts/`](../run_scripts/) take their settings from variables, which can be overridden when submitting a job, e.g. `VLLM_ENV=<path> sbatch run_scripts/serve_vllm_both_h200.sbatch`.

| Variable | Default | Used in | Meaning |
|---|---|---|---|
| `PROJECTROOT` | `$HOME/_PROJECTS/CI-impacts-information-retrieval` | serve and smoke-test scripts | project folder |
| `VLLM_ENV` | `$HOME/_PROJECTS/vllm/.venv` | all scripts | separate venv with vLLM installed (not the project venv) |
| `PYTHON_ENV` | `$PROJECTROOT/.venv` | serve and smoke-test scripts | project venv that runs the pipeline |
| `SCRIPT` | `examples/extract_ci_impacts.py` | serve scripts | pipeline script, relative to `PROJECTROOT` |
| `VLLM_VERSION` | `0.19.1` | `install_vllm.sbatch` | vLLM version to install |
| `REINSTALL` | `0` | `install_vllm.sbatch` | `1` recreates the vLLM venv |
| `UV_CACHE_DIR` | `$HOME/.cache/uv` | `install_vllm.sbatch` | cache folder of `uv` |

Additionally, adapt the `#SBATCH` header of the scripts (partition, account, GPU type, `--output`/`--error` log paths) to your cluster.

## Example-script settings
At the top of [`examples/extract_ci_impacts.py`](../examples/extract_ci_impacts.py):

| Setting | Default | Meaning |
|---|---|---|
| `SAMPLE_DOCUMENTS` | `None` | `None`: all documents, or a list of document names, e.g. `["Koks 2022 - Brief communication"]` |
| `PROCESS_HTML` | `False` | `True`: also fetch and clean the web pages of `text_sources_html.csv` |

## Web demo (Hugging Face Space)
The web demo [`hf_space/app.py`](../hf_space/app.py) reads [`configs/config_space.yaml`](../configs/config_space.yaml)
(selected via `CI_IMPACTS_CONFIG`): temporary paths under `/tmp/ci_impacts/` and the LLMs via Hugging Face Inference Providers
(`https://router.huggingface.co/v1`). The Space variables and secrets are listed in [`hf_space/README.md`](../hf_space/README.md):

| Setting | Default | Meaning |
|---|---|---|
| `LLM_EXTRACTOR_MODEL`, `JUDGE_MODEL` | `openai/gpt-oss-20b`, `google/gemma-4-31B-it` | models on Inference Providers |
| `ALLOW_SPACE_TOKEN` | `0` | `1`: visitors who are not signed in use the Space secret `HF_TOKEN` (billed to the Space owner) |
| `MAX_PDF_PAGES`, `MAX_CHUNKS` | `15`, `30` | limits per run |
| `LLM_EXTRACTOR_BASE_URL`, `JUDGE_BASE_URL`, `VLLM_API_KEY` | HF router | own vLLM servers instead of Inference Providers |

Local run (uses the environment variable `HF_TOKEN` if you are not logged in with `hf auth login`):
```bash
HF_TOKEN=<your token> uv run --with "gradio[oauth]" python hf_space/app.py
```
Deployment: push to the branch `hf-space` runs [`.github/workflows/deploy_space.yml`](../.github/workflows/deploy_space.yml)
(repository secret `HF_DEPLOY_TOKEN`, repository variable `HF_SPACE_ID`).

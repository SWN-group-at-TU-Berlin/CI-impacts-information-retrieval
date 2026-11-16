# CI-impacts-information-retrieval
Feasibility study to test the ability of LLMs to extract information about critical infrastructure (CI) impacts from text sources.

The pipeline extracts the different types of impacts of critical infrastructure failures from PDF documents and web pages, ranging from structural damages, over cascading effects to other CI sectors, to societal, economic and environmental impacts:

1. **Document processing** – PDFs (OCR with Docling/EasyOCR) and web pages are cleaned and translated to English (`src/document_cleaning.py`)
2. **Chunking + NER pre-filter** – only text chunks mentioning infrastructure are passed to the LLMs (spaCy, `src/tools/ner.py`)
3. **LLM 1** – LangGraph workflow with gpt-oss-20b: infrastructure cases and their cascading impacts (`src/models/`)
4. **Postprocessing** – grouping of the CI types, cleaning of the locations (`src/postprocess.py`)
5. **LLM-as-a-judge** – gemma-4-31B-it checks every extracted record against the source text (`src/models/llm_judge.py`)

The `main` branch always contains an applicable version of the project; the `dev` branch the most recent changes (this branch may not always provide an applicable version); the feature-branches represent the current working states. The feature branches history may change due to the use of `git rebase`, so it is better not to clone them.


## Requirements
- Virtual environment for Python 3.12 and [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- For the LLMs: GPU(s) with at least 80 GB GPU memory in total (e.g. 1x H200 or 2x A100) and CUDA version >=12.8. . These GPU settings are needed because the two LLMs used for extracting information are served with [vLLM](https://docs.vllm.ai/). Please note that there are some further small models which need some additional disk space (<5 GB): Docling (about 300 MB), spaCy (about 500 MB), currently five translation models (each  ~600 MB)
- The scripts in `run_scripts/` are written for an HPC cluster using Slurm.


## Usage – step by step

### 1. Install the project
```bash
git clone https://github.com/SWN-group-at-TU-Berlin/CI-impacts-information-retrieval.git
cd CI-impacts-information-retrieval

# create venv and install dependencies, activate venv
uv sync
source .venv/bin/activate
```
`uv sync` creates the virtual environment `.venv` with all packages of `pyproject.toml` (incl. the development tools, use `uv sync --no-dev` without them).
Run all commands with `uv run ...`.

### 2. Set your paths and servers in `configs/config.yaml`
All user settings are stored in [`configs/config.yaml`](configs/config.yaml) and read by `src/settings.py`. Adapt the entries to your machine:

| Entry | Default | Meaning |
|---|---|---|
| `PATH_DATA` | `../data` | input documents and cleaned documents (see step 4) |
| `PATH_RESULTS` | `../results` | extracted records of the pipeline |
| `PATH_LOGS` | `../logs` | log files |
| `DOCLING_ARTIFACTS` | `../docling_artifacts` | Docling models for the PDF processing (see step 5) |
| `SPACY_MODEL_PIPELINE` | `spacy_model_pipeline` | spaCy pipeline of the NER pre-filter (see step 6) |
| `HF_CACHE_DIR` | `null` | cache of the translation models, `null` = Hugging Face default (`~/.cache/huggingface/hub`) |
| `LLM1_BASE_URL`, `JUDGE_BASE_URL` | `http://localhost:8000/v1`, `http://localhost:8001/v1` | URLs of the two vLLM servers (see step 3) |
| `LLM1_MODEL`, `JUDGE_MODEL` | `openai/gpt-oss-20b`, `google/gemma-4-31B-it` | served model names |

- **Relative paths** are resolved from the project root (the folder with `pyproject.toml`), so `../data` is the folder `data` next to the project folder. Absolute paths (e.g. `/scratch/<user>/data`) and `~` work as well.
- **Overwriting single entries**: instead of editing `config.yaml`, single entries can also be set in the `.env` file, e.g. `PATH_DATA=/scratch/<user>/data`. A different config file can be used with `export CI_IMPACTS_CONFIG=configs/<your_config>.yaml`.
- **Priority**: environment variables > `.env` > `configs/config.yaml`.

Secrets do not belong into the config file. Copy the template and fill in your values:
```bash
cp .env_example .env      # .env is ignored by git
```

| `.env` entry | Needed for |
|---|---|
| `VLLM_API_KEY` | access to the vLLM servers (see step 3) |
| `HF_TOKEN` (optional) | download of gated models from Hugging Face, e.g. gemma for the judge on the GPU machine ([create a token](https://huggingface.co/docs/hub/security-tokens) and accept the license on the [model page](https://huggingface.co/google/gemma-4-31B-it)) |
| `HF_HOME` (optional) | cache folder of the LLM weights on the GPU machine, default `~/.cache/huggingface` |

### 3. Start the LLM servers (vLLM) and create the API key
The extraction LLM and the judge run as two OpenAI-compatible [vLLM servers](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html), protected by an API key.

1. **Set an API key**: vLLM does not issue keys, the key is a secret string that you choose yourself and that is passed to the server with `vllm serve --api-key <key>` (see the [vLLM docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html)). Choose a long key that is hard to guess and enter it in `.env`:
   ```bash
   VLLM_API_KEY=<your_key>
   ```
   The scripts in `run_scripts/` start the servers with this key and the pipeline sends the same key to the servers. Do not commit the key (`.env` is ignored by git).

2. **Install vLLM**
in a separate environment on the GPU machine (vLLM pins its own torch/CUDA versions), see the [vLLM installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/).
Gemma 4 needs vLLM >= 0.19.1 (thus the requirement for CUDA>=12.8) .

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

### 4. Add your documents
Store the documents in `<PATH_DATA>/text_sources/`:
- **PDFs**: named `<Authors> <Year> - <Title>.pdf`, e.g. `Koks 2022 - Brief communication.pdf`. The citation of the extracted records (e.g. `Koks 2022`) is taken from the file name.
- **Web pages** (optional): a table `text_sources_html.csv` with the columns `Authors`, `Year`, `Title`, `URL`. The pages are downloaded and cleaned by the pipeline.

The cleaned documents are automatically stored in `<PATH_DATA>/parsed_documents/` and reused in later runs.

### 5. Docling models (`DOCLING_ARTIFACTS`)
The PDF processing (Docling with OCR) loads its models from the folder `DOCLING_ARTIFACTS` and does **not** download missing models by itself. Download the layout model and the EasyOCR models once (see [Docling: model prefetching](https://docling-project.github.io/docling/usage/advanced_options/#model-prefetching-and-offline-usage)):
```bash
uv run docling-tools models download layout easyocr -o ../docling_artifacts   # use the path of DOCLING_ARTIFACTS
```
The folder then contains the layout model (`docling-project--docling-layout-heron`) and the folder `EasyOcr/` with the text detection model (`craft_mlt_25k.pth`) and the recognition models for English and Latin-script languages (`english_g2.pth`, `latin_g2.pth`).

### 6. spaCy model pipeline (`SPACY_MODEL_PIPELINE`)
The NER pre-filter uses the spaCy pipeline [`en_core_web_trf`](https://spacy.io/models/en#en_core_web_trf) (English, transformer-based) extended by the components `merge_entities` and `merge_noun_chunks`. The CI patterns (`ner_patterns.jsonl`) are added when the pipeline is loaded.

The pipeline is **created automatically** on the first run if the folder `SPACY_MODEL_PIPELINE` does not exist (download of about 460 MB). To create it beforehand, run
```bash
uv run python -c "from src.tools.ner import create_spacy_model_pipeline; create_spacy_model_pipeline()"
```
or create it manually:
```bash
uv run python -m spacy download en_core_web_trf
uv run python -c "
import spacy
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('merge_entities')
nlp.add_pipe('merge_noun_chunks')
nlp.to_disk('spacy_model_pipeline')   # path of SPACY_MODEL_PIPELINE
"
```

### 7. Translation models
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

### 8. Run the pipeline
The pipeline runs on the GPU node: the scripts of step 3 start both vLLM servers and then run [`examples/extract_ci_impacts.py`](examples/extract_ci_impacts.py) in the same job:
```bash
# Example:
sbatch run_scripts/serve_vllm_both_h200.sbatch
```
To run another script, pass it with `SCRIPT=<script> sbatch ...` (however currently a change of the script is less needed as only one example for the CI-extraction pipeline exists: [`examples/extract_ci_impacts.py`](examples/extract_ci_impacts.py) ).
In the settings at the top of the example you can restrict the run to a few documents (`SAMPLE_DOCUMENTS`) and switch on the processing of web pages (`PROCESS_HTML`).
<!-- The notebook [`examples/extract_ci_impacts.ipynb`](examples/extract_ci_impacts.ipynb) shows the same steps one by one; run it on the GPU node while the servers are running. -->

Results in `PATH_RESULTS`:
- `llm_chain_judge_<citation>.csv` – one row per extracted infrastructure case (type, amount, name, location, damage, cascading impacts) with the verdict of the judge
- `llm_responses_errors.csv` – chunks for which the LLM call failed

Logs are written to `PATH_LOGS`.


## Development

### Tests
```bash
uv run pytest tests/
```
The tests use mocked LLMs and a mocked spaCy model, so they run without GPUs and model downloads (also in the GitHub Actions CI).

### Pre-commit hooks
In case you have not used `pre-commit` hooks before, please run following lines just once when you set up the project locally
```bash
uv run pre-commit install
uv run pre-commit run --all-files
```
Every time you commit, the hooks are applied and help to maintain the quality of your code, e.g. by formatting and linting the code with `ruff` (remove whitespaces, consistent usage of quotes, sorted imports etc.).
You can run these hooks also manually by executing `uv run pre-commit run --all-files`. Note: the first run will take a while as the hooks are installed and cached.

### Vector database (not part of the pipeline in this release)
We aim to test different multilingual embedding models based on the suggestions from [HuggingFace Embedding Leaderboard](https://huggingface.co/spaces/mteb/leaderboard). However, we apply first open-source, lightweight models, such as from Ollama, to setup the pipeline and for creating a benchmark.\
To facilitate the generation and management of the embedding vectors stored in the postgres DB, along with the original text sources, we are making use of the [`pgai` package](https://github.com/timescale/pgai/tree/main/docs). Its implementation enables to use a postgres DB efficiently for Retrieval-Augmented Generation (RAG). However, `pgai` should be executed within a docker container. We create the database in the container by making use of the [timescale DB](https://hub.docker.com/r/timescale/timescaledb), an open-source database designed to make SQL scalable for large, (time-series) data and complex queries. \
The creation of the database within the container and the creation of the embeddings by using a vectorizer (i.e., embedding model), is done by following the steps described [here](https://github.com/timescale/pgai/blob/main/docs/vectorizer/quick-start.md). Make sure to have at least 15GB of free disk space and of course you need to have Docker installed.\
**Note**: Download the [pgai extension](https://github.com/timescale/pgai/tree/main/projects/extension) via `git clone` in your project folder, before you connect to the DB located in the docker container (see, step `docker compose exec -it db psql` mentioned [here](https://github.com/timescale/pgai/blob/main/docs/vectorizer/quick-start.md))


## Trouble shooting

- `401 Unauthorized` from the LLM servers: `VLLM_API_KEY` in `.env` differs from the key the servers were started with.
- `Connection refused` from the LLM servers: the servers are not running yet (startup can take up to 30 min), check `LLM1_BASE_URL` / `JUDGE_BASE_URL` and the job logs (`--output` / `--error` of the `#SBATCH` header, default `../logs/`).
- Docling cannot find a model (e.g. `EasyOcr`): download the models into `DOCLING_ARTIFACTS`, see step 5.
- Issues when installing `unstructured` -->  probably `rust` installation is missing:
  * First try to install `rustup` via `curl`: `$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` , See for more info [here](https://rust-lang.org/tools/install/)
  * Then install `unstructured` with an extension for all document types: `uv add unstructured[all-docs]`


## References

*Koks, E. E., van Ginkel, K. C. H., van Marle, M. J. E., and Lemnitzer, A.: Brief communication: Critical infrastructure impacts of the 2021 mid-July western European flood event, Nat. Hazards Earth Syst. Sci., 22, 3831–3838, https://doi.org/10.5194/nhess-22-3831-2022, 2022.*

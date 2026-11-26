# CI-impacts-information-retrieval
Feasibility study to test the ability of LLMs to extract information about critical infrastructure (CI) impacts from text sources.

The pipeline extracts the different types of impacts of critical infrastructure failures from PDF documents and web pages, ranging from structural damages, over cascading effects to other CI sectors, to societal, economic and environmental impacts:

1. **Document processing** – PDFs (OCR with Docling/EasyOCR) and web pages are cleaned and translated to English (`src/document_cleaning.py`)
2. **Chunking + NER pre-filter** – only text chunks mentioning infrastructure are passed to the LLMs (spaCy, `src/tools/ner.py`)
3. **LLM-extractor** – LangGraph workflow with gpt-oss-20b: infrastructure cases and their cascading impacts (`src/models/`)
4. **Postprocessing** – grouping of the CI types, cleaning of the locations (`src/postprocess.py`)
5. **LLM-as-a-judge** – gemma-4-31B-it checks every extracted record against the source text (`src/models/llm_judge.py`)

The extracted features (columns of the result tables) and their allowed values are listed in [docs/extracted_features.md](docs/extracted_features.md).

The `main` branch always contains an applicable version of the project; the `dev` branch the most recent changes (this branch may not always provide an applicable version); the feature-branches represent the current working states. The feature branches history may change due to the use of `git rebase`, so it is better not to clone them.


## Requirements
- Virtual environment for Python 3.12 and [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- For the LLMs: GPU(s) with at least 80 GB GPU memory in total (e.g. 1x H200 or 2x A100) and CUDA version >=12.8. . These GPU settings are needed because the two LLMs used for extracting information are served with [vLLM](https://docs.vllm.ai/). Please note that there are some further small models which need some additional disk space (<5 GB): Docling (about 300 MB), spaCy (about 500 MB), currently five translation models (each  ~600 MB)
- The scripts in `run_scripts/` are written for an HPC cluster using Slurm.


## Quickstart
```bash
git clone https://github.com/SWN-group-at-TU-Berlin/CI-impacts-information-retrieval.git
cd CI-impacts-information-retrieval
uv sync                                   # create .venv and install dependencies
cp .env_example .env                      # set VLLM_API_KEY, adapt configs/config.yaml
uv run docling-tools models download layout easyocr -o ../docling_artifacts
# spaCy pipeline: created automatically on the first run (needs internet on the node), or manually, see docs/installation.md step 4
sbatch run_scripts/install_vllm.sbatch    # once: vLLM in a separate venv on the GPU node
sbatch run_scripts/serve_vllm_both_h200.sbatch   # start both LLM servers and run the pipeline
```
Each step is explained in [docs/installation.md](docs/installation.md).


## Documentation
| Page | Content |
|---|---|
| [Installation and usage](docs/installation.md) | step-by-step setup: project, models, vLLM servers on an HPC cluster, adding documents, running the pipeline |
| [Configuration](docs/configuration.md) | all settings: `configs/config.yaml`, `.env`, run-script variables, example-script settings |
| [Extracted features](docs/extracted_features.md) | columns of the result tables, their description, type and allowed values |
| [Troubleshooting](docs/troubleshooting.md) | common errors and their solutions |
| [Contributing](CONTRIBUTING.md) | feedback, development setup, tests and pre-commit hooks |


## References

*Koks, E. E., van Ginkel, K. C. H., van Marle, M. J. E., and Lemnitzer, A.: Brief communication: Critical infrastructure impacts of the 2021 mid-July western European flood event, Nat. Hazards Earth Syst. Sci., 22, 3831–3838, https://doi.org/10.5194/nhess-22-3831-2022, 2022.*

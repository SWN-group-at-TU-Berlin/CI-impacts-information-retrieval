

# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]
### Changed
- "LLM 1" is now called **LLM-extractor** (docs, docstrings, log messages and the judge prompt)
- **Breaking:** the settings `LLM1_BASE_URL` and `LLM1_MODEL` are renamed to `LLM_EXTRACTOR_BASE_URL` and `LLM_EXTRACTOR_MODEL`;
  rename them in your `.env` / own config files
- README split into `docs/` (installation, configuration, extracted features, troubleshooting); the feature tables
  in `docs/extracted_features.md` are generated from the extraction schemas (`docs/generate_extracted_features.py`)

<!-- ### Added -->
<!-- - Re-implement developed postgres database for document storage and RAG -->


## [0.1.0] - 2026-11-16
First public release: a software project for extracting damage on critical infrastructure (CI) and cascading impacts from text sources with LLMs, tested on around 300 manually extracted CI-damage cases, commit history since Nov. 2025

### Added
- Document processing (`src/document_cleaning.py`): `PdfProcess` (Docling with EasyOCR) and `HtmlProcess`
  (web pages listed in `text_sources_html.csv`), text cleaning, language detection, and translation to English
  with Helsinki-NLP models (`src/models/llm_translate.py`)
- Chunking with Docling `HybridChunker` and an NER pre-filter (spaCy + CI entity patterns, `src/tools/ner.py`):
  only chunks that mention infrastructure are passed to the LLMs
- LLM extraction (`LlmExtract`, gpt-oss-20b) as a LangGraph workflow (`src/models/graph.py`):
  - direct impacts: infrastructure type, amount, name, location and damage category
  - cascading impacts: economic, societal, environmental and secondary infrastructure impacts
- LLM-as-a-judge (`LlmJudge`, gemma-4-31B-it) that checks each extracted record against the source text,
  using the same rules as the extraction LLM
- Postprocessing (`src/postprocess.py`): grouping of CI types into subgroups and sectors, mapping of damages to
  categories, cleaning of locations
- End-to-end pipeline `CiImpactPipeline` (`src/pipeline.py`) and an example (`examples/extract_ci_impacts.*`)
- Evaluation against a manually labelled dataset (precision, recall, F1, paragraph-wise matching) and figures
  (precision/recall bar plots, Sankey plot of cascading impacts)
- vLLM server scripts for 1 or 2 GPU nodes on the HPC cluster (`run_scripts/serve_vllm_*.sbatch`)
- Unit tests (pytest), pre-commit hooks (ruff, pre-commit-hooks) and GitHub Actions CI (actionlint, build, tests)

<!-- -  LLM extraction workflow publicly accessible;
- Init and extend workflow comprising:
        * LLM for extracting general infrastructure impacts, location and damage type (LLM 1) using tool
        * Standalone LLM for enriching locations (GeoLLM) [ initialized in the beginning but removed later, will be te-implemented in an upcoming release]
        * LLM-as-a-judge to recheck LLM1 response -->

### Changed
- LLMs are severed by vLLM instead of being called through the Nvidia NIM API. The latter caused increasingly TimeOutErrors due to heavy API-usage. Running them requires GPU node (s) with at least 80 GB of memory
- License changed from CC-BY-NC-ND to a stricter version AGPL v3 (i.e. modified versions, including those offered as a network service, must be published under the AGPL or GPL license)

### Removed
- Standalone location-enrichment LLM (GeoLLM): it hallucinated coordinates taken from other entries and could not geolocate areal features such as rivers or valleys



[Unreleased]: https://github.com/SWN-group-at-TU-Berlin/CI-impacts-information-retrieval/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SWN-group-at-TU-Berlin/CI-impacts-information-retrieval/releases/tag/v0.1.0

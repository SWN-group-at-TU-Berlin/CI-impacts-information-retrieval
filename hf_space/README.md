---
title: CI impacts extraction
emoji: 🌊
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.17.3
python_version: "3.12"
app_file: hf_space/app.py
pinned: false
license: agpl-3.0
short_description: Extract critical infrastructure impacts from text with LLMs
hf_oauth: true
hf_oauth_scopes:
  - inference-api
---

# CI impacts extraction – web demo

Paste a text or upload a PDF about a natural hazard event. The pipeline extracts the damaged critical infrastructure
and its cascading impacts. Results come as a small SQLite database and as a csv.
Source code and documentation: https://github.com/SWN-group-at-TU-Berlin/CI-impacts-information-retrieval

This file is the README of the **Space repository**. It is copied there by `.github/workflows/deploy_space.yml`.

## Who pays for the LLM calls?
The LLMs run on [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/pricing).
Visitors **sign in with Hugging Face**, and the calls use *their* token and monthly quota (scope `inference-api`).
The Space owner is not charged. Free accounts have a small monthly credit. When it is used up, requests fail until
next month (no bill).

## Space settings (Settings → Variables and secrets)
| Name | Kind | Default | Purpose |
|---|---|---|---|
| `LLM_EXTRACTOR_MODEL` | variable | `openai/gpt-oss-20b` | model of the LLM-extractor |
| `JUDGE_MODEL` | variable | `google/gemma-4-31B-it` | model of the LLM-as-a-judge, change it if no provider serves it |
| `ALLOW_SPACE_TOKEN` | variable | `0` | `1`: visitors who are not signed in use `HF_TOKEN` (billed to the owner) |
| `HF_TOKEN` | secret | – | token of the owner, only used with `ALLOW_SPACE_TOKEN=1` |
| `LLM_EXTRACTOR_BASE_URL`, `JUDGE_BASE_URL`, `VLLM_API_KEY` | secrets | HF router | use your own vLLM servers instead of Inference Providers |
| `MAX_PDF_PAGES`, `MAX_CHUNKS` | variables | `15`, `30` | limits per run |

Hardware: the free *CPU basic* hardware is enough. The LLMs run elsewhere; the Space only runs OCR, NER and translation.

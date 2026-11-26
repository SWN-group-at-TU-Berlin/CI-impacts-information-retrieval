# Troubleshooting

- `401 Unauthorized` from the LLM servers: `VLLM_API_KEY` in `.env` differs from the key the servers were started with.
- `Connection refused` from the LLM servers: the servers are not running yet (startup can take up to 30 min), check `LLM_EXTRACTOR_BASE_URL` / `JUDGE_BASE_URL` and the job logs (`--output` / `--error` of the `#SBATCH` header, default `../logs/`).
- Docling cannot find a model (e.g. `EasyOcr`): download the models into `DOCLING_ARTIFACTS`, see [step 3 of the installation](installation.md#3-docling-models-docling_artifacts).
- Issues when installing `unstructured` -->  probably `rust` installation is missing:
  * First try to install `rustup` via `curl`: `$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` , See for more info [here](https://rust-lang.org/tools/install/)
  * Then install `unstructured` with an extension for all document types: `uv add unstructured[all-docs]`

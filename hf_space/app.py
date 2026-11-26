#!/usr/bin/env python
"""Web demo (Gradio, Hugging Face Space) of the CI impact extraction pipeline.

Paste a text or upload a PDF -> document cleaning -> chunking -> NER pre-filter -> LLM-extractor -> postprocessing
-> LLM-as-a-judge (optional) -> results as SQLite database (src/results_db.py) and csv.

The LLMs are called via an OpenAI-compatible API, by default Hugging Face Inference Providers with the token of the
signed-in visitor (see configs/config_space.yaml and hf_space/README.md).

run locally from the project root:
    HF_TOKEN=<your token> uv run python hf_space/app.py
"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import os
import re
import shutil
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("CI_IMPACTS_CONFIG", str(PROJECT_ROOT / "configs" / "config_space.yaml"))

# ZeroGPU: `spaces` must be imported before torch / CUDA are initialized. Only the functions decorated with
# @spaces.GPU get a GPU (for a limited time per call); everything else, incl. the LLM API calls, runs on CPU.
# On CPU hardware and locally the decorator does nothing.
try:
    import spaces
except ImportError:  # local run without the `spaces` package
    spaces = None


def gpu(duration: int):
    return spaces.GPU(duration=duration) if spaces is not None else (lambda fn: fn)


import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

import src.results_db as rdb  # noqa: E402
from src.settings import settings as s  # noqa: E402

EXAMPLE_DB = PROJECT_ROOT / "examples" / "example_results.sqlite"
EXAMPLE_TEXT = (PROJECT_ROOT / "examples" / "example_input.txt").read_text()
FEATURES_DOC = PROJECT_ROOT / "docs" / "extracted_features.md"
HF_ROUTER = "router.huggingface.co"

MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "15"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "30"))  # upper limit of the slider: chunks sent to the LLMs per run
GPU_DURATION = int(os.getenv("GPU_DURATION", "120"))  # ZeroGPU: max. seconds of GPU time for document cleaning + NER per run
# True: anonymous visitors use the Space secret HF_TOKEN (billed to the Space owner). Always True when run locally.
ALLOW_SPACE_TOKEN = os.getenv("ALLOW_SPACE_TOKEN", "0") == "1" or "SPACE_ID" not in os.environ


def oauth_available() -> bool:
    """Sign-in with Hugging Face: on a Space with hf_oauth, locally only if logged in (Gradio mocks the login with the local token)"""
    if "SPACE_ID" in os.environ:
        return True
    from huggingface_hub import get_token

    return get_token() is not None


RESULT_COLUMNS = [
    "citation_id",
    "chunk_id",
    "infrastructure_type",
    "infrastructure_group",
    "location",
    "damage",
    "economic_impact_type",
    "economic_impact_amount",
    "societal_impact_type",
    "societal_impact_amount",
    "environmental_impact_type",
    "secondary_infrastructure_asset",
    "judge_verdict",
]


# --- Models, loaded once on first use ------------------------------------------------------------------------------

_chunker = None


def get_chunker():
    """Same chunker as in examples/extract_ci_impacts.py"""
    global _chunker
    if _chunker is None:
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer

        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
            max_tokens=256,
        )
        _chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    return _chunker


def ensure_docling_models() -> None:
    """Download the Docling models (layout, EasyOCR) for the PDF processing, if not yet done"""
    path = Path(s.DOCLING_ARTIFACTS)
    if path.exists() and any(path.iterdir()):
        return
    from docling.utils.model_downloader import download_models

    download_models(
        output_dir=path,
        with_layout=True,
        with_easyocr=True,
        with_tableformer=False,
        with_code_formula=False,
        with_picture_classifier=False,
        with_rapidocr=False,
    )


def build_models(api_key: str, use_judge: bool):
    """LLM-extractor (incl. LangGraph workflow) and LLM-as-a-judge, as in examples/extract_ci_impacts.py"""
    import src.models.graph as lg
    import src.models.llm_extract as le
    import src.models.llm_judge as lj
    import src.models.prompts as pr
    from src.models.schemas import JUDGE_FIELD_SCHEMA
    from src.tools.ner import get_infrastructure_entities

    llm_extract = le.LlmExtract(
        model=s.LLM_EXTRACTOR_MODEL,
        base_url=s.LLM_EXTRACTOR_BASE_URL,
        api_key=api_key,
        tools=[get_infrastructure_entities],
        max_tokens=8192,
        reasoning_effort="low",
    )
    lg.build_graph(llm_extract)
    llm_judge = None
    if use_judge:
        llm_judge = lj.LlmJudge(
            model=s.JUDGE_MODEL,
            base_url=s.JUDGE_BASE_URL,
            api_key=api_key,
            prompt=pr.judge_prompt,
            field_schema=JUDGE_FIELD_SCHEMA,
            extra_rules=pr.JUDGE_EXTRA_RULES,
            ner_fn=lambda text: get_infrastructure_entities.invoke({"text": text}) or [],
            batch_size=5,
            max_tokens=8000,
        )
    return llm_extract, llm_judge


def resolve_api_key(oauth_token: gr.OAuthToken | None) -> str | None:
    """API key of the LLM endpoints: own vLLM servers -> VLLM_API_KEY, HF Inference Providers -> token of the visitor"""
    if HF_ROUTER not in s.LLM_EXTRACTOR_BASE_URL:
        return s.VLLM_API_KEY
    if oauth_token is not None:
        return oauth_token.token
    return os.getenv("HF_TOKEN") if ALLOW_SPACE_TOKEN else None


def api_error_hint(error: str) -> str | None:
    """Explanation of LLM API errors that would occur for every chunk (authentication, quota, model), else None"""
    hints = {
        "401": "the API key was rejected. Sign out and sign in with Hugging Face again.",
        "402": "your monthly Inference Providers credit is used up (free accounts: small monthly quota).",
        "403": "your Hugging Face login/token is not allowed to call Inference Providers. Sign out and sign in again and "
        "allow the 'inference' permission; a token in HF_TOKEN needs the permission 'Make calls to Inference Providers'.",
        "404": f"the model is not available on Inference Providers: check LLM_EXTRACTOR_MODEL ({s.LLM_EXTRACTOR_MODEL}) and JUDGE_MODEL ({s.JUDGE_MODEL}).",
        "429": "too many requests or quota exceeded. Try again later.",
    }
    return next((f"LLM API error {code}: {hint}" for code, hint in hints.items() if f"Error code: {code}" in error), None)


# --- Documents -----------------------------------------------------------------------------------------------------


def text_to_document(text: str, name: str, output_dir: Path):
    """Cleaned DoclingDocument of a pasted text, with the same cleaning and translation steps as for the PDFs
    (methods of src.document_cleaning._DocProcessing). Paragraphs are separated by blank lines.
    """
    import src.document_cleaning as dc

    processor = dc.PdfProcess(output_dir=output_dir, overwrite=True)
    language = processor.detect_language(text[:2000])
    paragraphs = [processor.clean_paragraph(p) for p in re.split(r"\n\s*\n", text) if p.strip()]
    paragraphs = [p for p in (processor.finalize_paragraph(p, language) for p in paragraphs if p) if p]
    return processor.save(name, paragraphs, ["Text"] * len(paragraphs)), language


def check_pdf(pdf_path: str) -> None:
    import pypdfium2

    n_pages = len(pypdfium2.PdfDocument(pdf_path))
    if n_pages > MAX_PDF_PAGES:
        raise gr.Error(f"The PDF has {n_pages} pages, the demo processes at most {MAX_PDF_PAGES}.")


def pdf_to_document(pdf_path: str, output_dir: Path):
    """Cleaned DoclingDocument of an uploaded PDF (OCR with Docling), its name '<authors> <year> - <title>' and language"""
    import src.document_cleaning as dc

    stem = Path(pdf_path).stem
    # the citation is taken from the file name "<authors> <year> - <title>", see dc.extract_citation_info
    if not re.search(r"\d{4}", stem):
        stem = f"Upload {date.today().year} - {stem}"
    filepath = output_dir / f"{stem}.pdf"
    shutil.copy(pdf_path, filepath)
    processor = dc.PdfProcess(output_dir=output_dir, overwrite=True)
    _, _, title = dc.extract_citation_info(stem)
    return processor.process(filepath), stem, processor.detect_language(title)


@gpu(duration=GPU_DURATION)
def prepare_chunks(text: str, pdf_file: str | None, workdir: str) -> dict:
    """Document cleaning (OCR, translation) -> chunking -> NER pre-filter. On ZeroGPU this step runs on a GPU.
    Returns only plain data (no DoclingDocument), as ZeroGPU passes the result from the GPU worker back to the app.
    """
    import src.pipeline as pl
    from src.tools.ner import get_infrastructure_entities

    if pdf_file is not None:
        doclingdoc, name, language = pdf_to_document(pdf_file, Path(workdir))
        source_type = "pdf"
    else:
        name = f"Text {date.today().year} - pasted text"
        doclingdoc, language = text_to_document(text, name, Path(workdir))
        source_type = "text"
    citation = pl.get_citation(name)

    chunk_meta = []
    for chunk_no, chunk in enumerate(get_chunker().chunk(dl_doc=doclingdoc)):
        chunk_text = pl.prepare_chunk_text(chunk.text)
        entities = get_infrastructure_entities.invoke({"text": chunk_text}) or []
        chunk_meta.append({"citation_id": citation, "chunk_id": chunk_no, "text": chunk_text, "ner_entities": entities, "passed_ner": bool(entities)})
    return {"name": name, "citation": citation, "language": language, "source_type": source_type, "chunks": chunk_meta}


# --- Pipeline run --------------------------------------------------------------------------------------------------


def run_pipeline(text: str, pdf_file, use_judge: bool, max_chunks: int, oauth_token: gr.OAuthToken | None = None):
    """Generator for the UI: yields (log, records table, db file, csv file, table names)"""
    import src.pipeline as pl

    log = []

    def update(df=None, db=None, csv=None):
        tables = gr.update(choices=rdb.list_tables(db), value="v_cases_flat") if db else gr.update()
        return "\n".join(log), df, db, csv, tables

    if not (text and text.strip()) and pdf_file is None:
        raise gr.Error("Paste a text or upload a PDF.")
    api_key = resolve_api_key(oauth_token)
    if not api_key:
        raise gr.Error("Please sign in with Hugging Face first: the LLMs run on your own Inference Providers quota.")

    workdir = Path(tempfile.mkdtemp(prefix="ci_impacts_"))
    started_at, time0 = time.strftime("%Y-%m-%dT%H:%M:%S"), time.time()

    # 1. document cleaning, 2. chunking, 3. NER pre-filter (on the GPU on ZeroGPU)
    if pdf_file is not None:
        check_pdf(pdf_file)
        ensure_docling_models()  # download on CPU, before the GPU time starts
        log.append(f"- Processing PDF `{Path(pdf_file).name}` (OCR, translation, NER; can take a few minutes) …")
    else:
        log.append("- Cleaning the text (translation, NER) …")
    yield update()
    prepared = prepare_chunks(text, pdf_file, str(workdir))
    name, citation, language, source_type = prepared["name"], prepared["citation"], prepared["language"], prepared["source_type"]
    chunks, chunk_meta = [c["text"] for c in prepared["chunks"]], prepared["chunks"]
    log.append(f"- Document `{citation}`, language `{language}`" + (" (translated to English)" if language != "en" else ""))
    to_llm = [c for c in chunk_meta if c["passed_ner"]][: int(max_chunks)]
    log.append(f"- {len(chunks)} chunks, {sum(c['passed_ner'] for c in chunk_meta)} mention infrastructure, {len(to_llm)} are sent to the LLMs")
    yield update()

    # 4. LLM-extractor -> postprocessing -> LLM-as-a-judge, per chunk
    llm_extract, llm_judge = build_models(api_key, use_judge)
    pipeline = pl.CiImpactPipeline(get_chunker(), llm_extract, llm_judge, retry_wait=10, verbose=False)
    records = []
    for i, chunk in enumerate(to_llm):
        log.append(f"- Chunk {i + 1}/{len(to_llm)}: {', '.join(chunk['ner_entities'][:5])} …")
        yield update(pd.concat(records, ignore_index=True)[[c for c in RESULT_COLUMNS if c in records[0]]] if records else None)
        n_errors = len(pipeline.errors)
        df = pipeline.extract_chunk(chunk["text"], citation, chunk["chunk_id"])
        if len(pipeline.errors) > n_errors:
            error = pipeline.errors[-1]["error"]
            log[-1] = log[-1].removesuffix(" …") + f" → ⚠ **LLM error:** `{error[:400]}`"
            hint = api_error_hint(error)
            if hint:  # the same error would occur for every chunk: stop instead of skipping all chunks
                log.append(f"- ⛔ **Stopped:** {hint}")
                gr.Warning(hint)
                break
        elif df is not None:
            records.append(df)
            log[-1] = log[-1].removesuffix(" …") + f" → {len(df)} record(s)"
        else:
            log[-1] = log[-1].removesuffix(" …") + " → no CI impacts"
    if not to_llm:
        log.append("- ⚠ No chunk mentions an infrastructure asset (NER pre-filter), so nothing was sent to the LLMs.")

    # 5. results database
    df_all = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    stem = re.sub(r"\W+", "_", citation)
    csv_path = workdir / f"llm_chain_judge_{stem}.csv"
    df_all.to_csv(csv_path, index=False)
    db_path = rdb.write_results_db(
        df_all,
        workdir / f"ci_impacts_{stem}.sqlite",
        documents=[{"citation_id": citation, "title": name, "source_type": source_type, "language": language}],
        chunks=chunk_meta,
        run={
            "extractor_model": s.LLM_EXTRACTOR_MODEL,
            "judge_model": s.JUDGE_MODEL if use_judge else None,
            "started_at": started_at,
            "duration_s": round(time.time() - time0, 1),
            "n_errors": len(pipeline.errors),
        },
        ci_patterns=pipeline.ci_patterns,
    )
    log.append(f"- **Done** in {time.time() - time0:.0f} s: {len(df_all)} infrastructure case(s)")
    df_show = df_all[[c for c in RESULT_COLUMNS if c in df_all.columns]] if not df_all.empty else pd.DataFrame(columns=RESULT_COLUMNS)
    yield update(df_show, str(db_path), str(csv_path))


def show_table(db_path, table):
    if not db_path or not table:
        return None
    return rdb.read_table(db_path, table)


# --- UI ------------------------------------------------------------------------------------------------------------

SCHEMA_MD = """
| Table | One row per | Key columns |
|---|---|---|
| `documents` | input document | `citation_id`, `source_type`, `language`, `n_chunks` |
| `chunks` | text chunk (max. 256 tokens) | `chunk_no`, `text`, `ner_entities`, `passed_ner` |
| `infrastructure_cases` | damaged infrastructure asset | `infrastructure_type`, `infrastructure_group`, `infrastructure_sector`, `location`, `damage`, `judge_verdict` |
| `economic_impacts` | economic impact of a case | `origin`, `type`, `amount`, `unit` |
| `societal_impacts` | societal impact of a case | `origin`, `type`, `amount`, `unit` |
| `environmental_impacts` | environmental impact of a case | `origin`, `type`, `amount`, `unit` |
| `secondary_infrastructure_impacts` | cascading effect on another asset | `target_asset`, `target_group`, `target_sector` |
| `judge_adaptations` | field changed by the LLM-as-a-judge | `field`, `old_value`, `new_value` |
| `runs` | pipeline run | `extractor_model`, `judge_model`, `duration_s` |
| `v_cases_flat` (view) | infrastructure case | same columns as the flat csv of the pipeline |

`documents 1─n chunks 1─n infrastructure_cases 1─n {economic, societal, environmental, secondary_infrastructure}_impacts, judge_adaptations`
"""

INTRO_MD = f"""
# Critical infrastructure impacts from text

Extracts damaged **critical infrastructure** and their **cascading impacts** (economic, societal, environmental,
other infrastructure) from a text or PDF about a natural hazard event, with two LLMs and a judge step.
Code and documentation: [GitHub](https://github.com/SWN-group-at-TU-Berlin/CI-impacts-information-retrieval).

**Costs:** the LLMs (`{s.LLM_EXTRACTOR_MODEL}`, judge `{s.JUDGE_MODEL}`) run on Hugging Face Inference Providers
with **your** account after you sign in. Free accounts have a small monthly quota; when it is used up, requests
fail until next month (nothing is billed). A short article uses roughly 10–20k tokens per chunk.
"""

with gr.Blocks(title="CI impacts extraction") as demo:
    gr.Markdown(INTRO_MD)
    with gr.Tab("Run pipeline"):
        if oauth_available():
            gr.LoginButton()
        else:
            gr.Markdown("*Running locally without Hugging Face login: the LLMs are called with the environment variable `HF_TOKEN`.*")
        with gr.Row():
            with gr.Column(scale=3):
                text_in = gr.Textbox(label="Text (paragraphs separated by blank lines)", lines=12, placeholder="Paste a report or news article …")
                example_btn = gr.Button("Load example text", size="sm")
            with gr.Column(scale=2):
                pdf_in = gr.File(label=f"…or upload a PDF (max. {MAX_PDF_PAGES} pages)", file_types=[".pdf"], type="filepath")
                judge_in = gr.Checkbox(value=True, label="Run LLM-as-a-judge")
                chunks_in = gr.Slider(1, MAX_CHUNKS, value=10, step=1, label="Max. chunks sent to the LLMs")
                run_btn = gr.Button("Run pipeline", variant="primary")
        log_out = gr.Markdown()
        records_out = gr.Dataframe(label="Extracted infrastructure cases", wrap=True)
        with gr.Row():
            db_out = gr.File(label="Results database (.sqlite)")
            csv_out = gr.File(label="Flat table (.csv)")
        with gr.Accordion("Browse the results database", open=False):
            table_in = gr.Dropdown(label="Table", choices=[])
            table_out = gr.Dataframe(wrap=True)

        example_btn.click(lambda: EXAMPLE_TEXT, outputs=text_in)
        run_btn.click(
            run_pipeline,
            inputs=[text_in, pdf_in, judge_in, chunks_in],
            outputs=[log_out, records_out, db_out, csv_out, table_in],
            concurrency_limit=1,  # the LangGraph workflow holds the LLM-extractor in a module global (src/models/graph.py)
        )
        table_in.change(show_table, inputs=[db_out, table_in], outputs=table_out)

    with gr.Tab("Example database"):
        gr.Markdown(
            "Output format of the pipeline for the example text (*Load example text*). The records of this example are "
            "**illustrative** (hand-written in the format of the LLMs, see `examples/build_example_db.py`), not a real model output."
        )
        gr.Markdown(SCHEMA_MD)
        example_db = gr.State(str(EXAMPLE_DB))
        example_table_in = gr.Dropdown(label="Table", choices=rdb.list_tables(EXAMPLE_DB), value="v_cases_flat")
        example_table_out = gr.Dataframe(value=rdb.read_table(EXAMPLE_DB, "v_cases_flat"), wrap=True)
        gr.File(value=str(EXAMPLE_DB), label="Download example database")
        example_table_in.change(show_table, inputs=[example_db, example_table_in], outputs=example_table_out)

    with gr.Tab("Extracted features"):
        gr.Markdown(FEATURES_DOC.read_text())


if __name__ == "__main__":
    demo.queue().launch()

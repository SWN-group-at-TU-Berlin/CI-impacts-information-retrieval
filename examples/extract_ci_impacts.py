# # CI impact extraction pipeline – example
#
# Runs the extraction pipeline of `notebooks/extract_ciimpacts.py` on a few documents:
#
# 1. **Document cleaning** – PDF (OCR) and HTML sources → cleaned DoclingDocuments (`src.document_cleaning`)
# 2. **Chunking** – HybridChunker (max. 256 tokens)
# 3. **NER pre-filter** – only chunks with infrastructure entities are passed to the LLMs (`src.tools.ner`)
# 4. **LLM-extractor** – LangGraph workflow (gpt-oss): infrastructure cases + cascading impacts (`src.models.llm_extract`, `src.models.graph`)
# 5. **Postprocessing** – grouping of the CI types, cleaning of the locations (`src.postprocess`)
# 6. **LLM-as-a-judge** – plausibility check of each record against the chunk text (`src.models.llm_judge`)
#
# Steps 3-6 per chunk and document are run by `CiImpactPipeline` (`src/pipeline.py`).
#
# Requirements: the vLLM servers of LLM-extractor and the judge are running and reachable (`LLM_EXTRACTOR_BASE_URL`, `JUDGE_BASE_URL`, `VLLM_API_KEY` in `.env`), see `run_scripts/serve_vllm_*.sbatch`.

# ## Setup
import os
import sys
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

import src.document_cleaning as dc
import src.models.graph as lg
import src.models.llm_extract as le
import src.models.llm_judge as lj
import src.models.prompts as pr
from src.models.schemas import JUDGE_FIELD_SCHEMA, InfrastructureCases
from src.pipeline import CiImpactPipeline
from src.settings import settings as s
from src.tools.ner import get_infrastructure_entities

SAMPLE_DOCUMENTS = None  # None: all documents, or a list of document names, e.g. ["Koks 2022 - Brief communication"]
PROCESS_HTML = False  # True: also fetch and clean the web pages of text_sources_html.csv


# run from the project root:
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

DOCS_DIR = Path(s.PATH_DATA / "text_sources")
PARSED_TEXT_DIR = Path(s.PATH_DATA / "parsed_documents")
OUTPUT_DIR = s.PATH_RESULTS
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ## 1. Document cleaning
# Already cleaned documents (`*_cleaned.md`) are loaded instead of processed again.

pdf_filepaths = [f for f in DOCS_DIR.glob("*.pdf") if SAMPLE_DOCUMENTS is None or f.stem in SAMPLE_DOCUMENTS]
documents = dc.PdfProcess(output_dir=PARSED_TEXT_DIR).process_all(pdf_filepaths)

if PROCESS_HTML:
    documents |= dc.HtmlProcess(
        Path(DOCS_DIR, "text_sources_html.csv"),
        output_dir=Path(PARSED_TEXT_DIR, "html_parsed"),
    ).process_all()

print(f"{len(documents)} documents: {list(documents)}")


# ## 2. Chunker

embed_model = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(embed_model),
    max_tokens=256,  # max tokens for MiniLM-l6-v2
    padding=True,
    truncation=True,
)
# hierarchical chunker which also considers the max. token length and merges smaller chunks (merge_peers=True)
chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

# ## 3. LLM-extractor (CI impact extraction) and LLM-as-a-judge

# LLM-extractor (gpt-oss) incl. the NER tool, the graph (nodes, prompts, output schemas) is defined in src/models/graph.py
llm_extract = le.LlmExtract(
    model=s.LLM_EXTRACTOR_MODEL,
    base_url=s.LLM_EXTRACTOR_BASE_URL,
    api_key=s.VLLM_API_KEY,
    tools=[get_infrastructure_entities],
    max_tokens=8192,  # reasoning + answer: prompt + max_tokens must be <= vLLM --max-model-len
    reasoning_effort="low",
)
graph = lg.build_graph(llm_extract)

# LLM-as-a-judge: uses the same field definitions (output schemas), keyword lists and NER entities as LLM-extractor
llm_judge = lj.LlmJudge(
    model=s.JUDGE_MODEL,
    base_url=s.JUDGE_BASE_URL,
    api_key=s.VLLM_API_KEY,
    prompt=pr.judge_prompt,
    field_schema=JUDGE_FIELD_SCHEMA,
    extra_rules=pr.JUDGE_EXTRA_RULES,
    ner_fn=lambda text: get_infrastructure_entities.invoke({"text": text}) or [],
    batch_size=5,
    max_tokens=8000,
)

# smoke test: both vLLM servers should list their served model
llm_extract.smoke_test(schema=InfrastructureCases)
llm_judge.smoke_test()

# graph  # structure of the LangGraph workflow

# ## 4. Pipeline
# Per chunk: NER pre-filter → LLM-extractor → grouping of the CI types + cleaning of the locations → LLM-as-a-judge, see `src/pipeline.py`.

pipeline = CiImpactPipeline(chunker, llm_extract, llm_judge, output_dir=OUTPUT_DIR)

# ## 5. Run
df_all = pipeline.run(documents)  # records of all documents, also saved per document in OUTPUT_DIR


# extracted records incl. the verdicts of the LLM-as-a-judge
columns = [
    "citation_id",
    "chunk_id",
    "infrastructure_type",
    "infrastructure_group",
    "location",
    "damage",
    "economic_impact_type",
    "societal_impact_type",
    "secondary_infrastructure_asset",
    "judge_verdict",
]
df_all[[c for c in columns if c in df_all.columns]]

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CI impact extraction pipeline per chunk and document: NER pre-filter -> LLM 1 (LlmExtract) -> postprocessing -> LLM-as-a-judge (LlmJudge)"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import re
import time
from pathlib import Path

import pandas as pd

import src.document_cleaning as dc
import src.postprocess as pp
import src.models.llm_extract as le
import src.models.llm_judge as lj
from src.tools.ner import get_infrastructure_entities, load_ci_patterns


def prepare_chunk_text(text: str) -> str:
    """Add the punctuation of "e.g." and "i.e." back for the LLMs, it is removed during the document cleaning
    (dc._DocProcessing.clean_paragraph: "e.g. " -> "eg ") to avoid sentence splits by the chunker"""
    text = re.sub(r"\beg\s+", "e.g. ", text)
    return re.sub(r"\bie\s+", "i.e. ", text)


def get_citation(name: str) -> str:
    """Citation id "<authors> <year>" of a document name "<authors> <year> - <title>" """
    author, year, _ = dc.extract_citation_info(name)
    return f"{author} {year}".replace("  ", " ").strip()


class CiImpactPipeline:
    """Extraction of CI impacts from cleaned documents (DoclingDocument, see dc.PdfProcess, dc.HtmlProcess).

    Per chunk: NER pre-filter (only chunks mentioning infrastructure are passed to the LLMs) -> LLM 1 (llm_extract)
    -> records -> grouping of the CI types (records which are not about CI are removed) and cleaning of the locations
    -> LLM-as-a-judge (llm_judge, optional).

    usage:
        pipeline = CiImpactPipeline(chunker, llm_extract, llm_judge, output_dir=OUTPUT_DIR)
        df_records = pipeline.run(documents)
    """

    def __init__(self, chunker, llm_extract: le.LlmExtract, llm_judge: lj.LlmJudge | None = None,
                 ci_patterns: pd.DataFrame | None = None, output_dir: Path | None = None,
                 max_retries: int = 1, retry_wait: int = 60, verbose: bool = True):
        self.chunker = chunker
        self.llm_extract = llm_extract
        self.llm_judge = llm_judge
        # CI types and their subgroups, for grouping the extracted infrastructure types
        self.ci_patterns = ci_patterns if ci_patterns is not None else load_ci_patterns()
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.verbose = verbose
        self.errors: list[dict] = []  # chunks skipped due to errors of LLM 1 (e.g. LengthLimitError)
        self.not_grouped = pd.DataFrame()  # records whose infrastructure type is no CI type

    def postprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Records about CI (grouped CI types, cleaned locations) and records whose infrastructure type matches no CI type, see pp.postprocess_response"""
        return pp.postprocess_response(df, self.ci_patterns)

    def extract_chunk(self, text: str, citation: str, chunk_no: int) -> pd.DataFrame | None:
        """Records of one chunk: LLM 1 -> postprocessing -> LLM-as-a-judge. None if the chunk contains no CI impacts."""
        # NER pre-filter: only chunks which mention infrastructure are passed to the LLMs
        if not get_infrastructure_entities.invoke({"text": text}):
            return None

        # LLM 1
        try:
            result = self.llm_extract.extract(text, max_retries=self.max_retries, retry_wait=self.retry_wait)
        except Exception as e:
            print(f"! LLM 1 failed for chunk {chunk_no} of {citation}: {type(e).__name__}: {e}")
            self.errors.append({"citation_id": citation, "chunk_id": chunk_no, "response": None, "error": f"{type(e).__name__}: {e}"})
            return None
        if not result.get("is_infrastructure_case"):
            return None

        try:
            df = le.result_to_dataframe(result)
        except Exception as e:
            print(f"! Cannot convert LLM 1 response of chunk {chunk_no} of {citation}: {e}")
            self.errors.append({"citation_id": citation, "chunk_id": chunk_no, "response": str(result), "error": f"{type(e).__name__}: {e}"})
            return None
        df["citation_id"], df["chunk_id"], df["chunk_text"] = citation, chunk_no, text
        if df.empty or df["infrastructure_type"].isna().all():
            return None

        df, not_grouped = self.postprocess(df)
        self.not_grouped = pd.concat([self.not_grouped, not_grouped], ignore_index=True)
        if df.empty:
            return None
        # NOTE no check if the amounts are mentioned in the text
        # df = pp.remove_amounts_not_in_text(df, "societal_impact_amount", "chunk_text")

        # LLM-as-a-judge
        if self.llm_judge is not None:
            df = self.llm_judge.judge(df, text)
            if self.verbose:
                print(lj.format_judge_report(df, verdicts="adapted"))  # verdict counts + adapted records (old -> new)
        return df

    def extract_document(self, name: str, doclingdoc) -> pd.DataFrame:
        """Records of all chunks of a cleaned document, saved as csv in output_dir"""
        citation = get_citation(name)
        chunks = list(self.chunker.chunk(dl_doc=doclingdoc))
        records = []
        for chunk_no, chunk in enumerate(chunks):
            if self.verbose:
                print(f"[{citation}] chunk {chunk_no + 1}/{len(chunks)}")
            df = self.extract_chunk(prepare_chunk_text(chunk.text), citation, chunk_no)
            if df is not None:
                records.append(df)
        df_doc = pd.concat(records, ignore_index=True) if records else pd.DataFrame()

        if self.output_dir is not None:
            df_doc.to_csv(self.output_dir / f"llm_chain_judge_{citation}.csv", encoding="utf-8", index=False)
            pd.DataFrame(self.errors, columns=["citation_id", "chunk_id", "response", "error"]).astype(str).to_csv(
                self.output_dir / "llm_responses_errors.csv", encoding="utf-8", index=False)
        return df_doc

    def run(self, documents: dict) -> pd.DataFrame:
        """Records of all documents ({name: DoclingDocument})"""
        time0 = time.time()
        results = []
        for doc_no, (name, doclingdoc) in enumerate(documents.items()):
            print(f"\n######## Processing document [{doc_no + 1}/{len(documents)}]: {name} ########")
            df_doc = self.extract_document(name, doclingdoc)
            print(f"{name}: {len(df_doc)} records")
            results.append(df_doc)
        print(f"CI impact extraction took {(time.time() - time0) / 60:.1f} min, {len(self.errors)} chunks skipped due to errors")
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

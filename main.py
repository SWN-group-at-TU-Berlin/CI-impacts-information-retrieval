#!/usr/bin/env python3

import os
import re
import time
from pathlib import Path

from pdfminer.high_level import extract_text
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.document_converter import DocumentConverter, FormatOption


import src.settings as s
from src.database import fill_db, TextSource
from src.document_cleaning import remove_references, remove_headers_footers


# Docling pipeline configs
accelerator_options = AcceleratorOptions(
    num_threads=4, device=AcceleratorDevice.AUTO
)  # use GPU + multi-threading
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = (
    True  # identify tables as such just not to have them in the TextItems later
)
pipeline_options.accelerator_options = accelerator_options
pipeline_options.force_backend_text = True


DOCS_DIR = "../" + s.settings.PATH_DATA + "text_sources/"
PARSED_TEXT_DIR = "../" + s.settings.PATH_DATA + "parsed_documents/"


md_dir = Path(PARSED_TEXT_DIR)
md_dir.mkdir(parents=True, exist_ok=True)


# logging.basicConfig(level=logging.INFO  # TODO  init logger

DOCS_DIR = "../" + s.settings.PATH_DATA + "text_sources/"


# POPULATE DATABASE
for filename in os.listdir(DOCS_DIR):
    if filename.endswith(".pdf"):
        print(f"fetching: {filename}")

        file_path = os.path.join(DOCS_DIR, filename)
        text = extract_text(file_path)
        filename = Path(filename).stem
        authors, title = authors, title = (
            re.compile(r"(.+?)[0-9]{4}(.*)?").search(filename).groups()
        )

        entry = {
            "authors": authors.strip(),
            "title": title.strip(),
            "source": "dummy source",
            "content": text,
            "metadata": {
                "tags": ["ahr_valley", "dummy_publication_type"],
                "published_date": re.findall(r"[0-9]{4}", filename)[0],
            },
        }
    fill_db(TextSource(**entry))


# DOCUMENT PARSING AND CLEANING

converted = DocumentConverter(
    allowed_formats=[InputFormat.PDF, InputFormat.MD],
    format_options={
        InputFormat.PDF: FormatOption(
            pipeline_cls=StandardPdfPipeline,
            pipeline_options=pipeline_options,
            backend=PyPdfiumDocumentBackend,
        ),
    },
)

for pdf_filename in os.listdir(DOCS_DIR):
    if pdf_filename.endswith(".pdf"):
        md_filename = f"{Path(pdf_filename).stem}.md"

        pdf_filepath = os.path.join(DOCS_DIR, Path(pdf_filename))
        md_filepath = os.path.join(PARSED_TEXT_DIR, Path(md_filename))
        cleaned_md_filepath = md_filepath.replace(".md", "_cleaned.md")

        if os.path.exists(md_filepath):
            print(
                f"Markdown file '{md_filepath}' already exists. Skipping conversion and cleaning."
            )
            continue

        start_time = time.time()
        print(f"\nFetching: {pdf_filename}")

        print("Remove reference section")
        pdf_text = extract_text(pdf_filepath)
        pdf_text_no_references = remove_references(pdf_text)

        # FIXME remove workaround of saving pdf as markdown and reading it again as Docling.Document
        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(pdf_text_no_references)

        # FIXME with DocLoader
        # loader = DoclingLoader(md_filepath)
        # md_text = loader.load()
        print("Converting Markdown to text...")
        md_text = converted.convert(md_filepath)

        print("Remove headers and footers")
        md_text_cleaned = remove_headers_footers(md_text)

        md_text_cleaned.document.save_as_markdown(cleaned_md_filepath)
        print(
            f"Parsed and cleaned document saved as markdown to: {cleaned_md_filepath}"
        )

        end_time = time.time() - start_time
        print(f"Parsing and cleaning done. Time elapsed: {end_time:.2f} seconds.")


## TODO init main function
def main():
    pass  # placeholder


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parse and clean text sources"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

from contextlib import suppress
import re
import warnings
from docling_core.types.doc.document import TextItem
from docling.document_converter import ConversionResult
from langchain_core.documents import Document
import json
from typing import Iterable

def extract_citation_info(citation_text: str) -> str:
    """Extract citation information from the document title (i.e. citation id)."""
    
    citation_pattern = r"(.*?)(\d{4})(.*)" # split at first occurrence of year
    # with suppress(AttributeError):
    authors, year, title = re.findall(citation_pattern, citation_text)[0]#[:2] 
    authors = authors.replace("et al ", "")

    return authors, year, title


def remove_urls(document_text: str) -> str:
    """Remove URLs from the document text."""
    url_pattern = r"http\S+|www\S+|https\S+"
    document_text_no_urls = re.sub(url_pattern, "", document_text, flags=re.MULTILINE) # find urls everywhere in text
    return document_text_no_urls


def remove_references(document_text: str) -> str:
    # search for reference section
    pattern = re.compile(
        r"^(References|REFERENCES|Bibliography|BIBLIOGRAPHY)$", flags=re.MULTILINE
    )
    # re.MULTILINE in combination with "^" and case sensitive : find search words only when they are at beginning of a new line
    matches = list(pattern.finditer(document_text))
    matches_list = [i.group() for i in matches]

    if len(matches_list) != 1:
        warnings.warn(
            f"""Expected one match, but found {len(matches_list)} matches,
                taking the last occurred match for determining the start of the reference section."""
        )
    if matches:
        last_match = matches[-1]
        start_index = last_match.end()
        document_text_no_references = document_text[:start_index].strip()
        return document_text_no_references
    else:
        print("No References section found!")
        return document_text


# def remove_headers_footers(conv_file: ConversionResult) -> ConversionResult:
#     ## remove headers and footers from the document
#     total_texts = len(conv_file.document.texts)
#     print(f"Total texts in document: {total_texts}")

#     text_items = [x for x in conv_file.document.texts if isinstance(x, TextItem)]

#     text_items_to_drop = []
#     text_items_to_drop_visualization = []

#     ## select text items to drop based on their number of chars, e.g. headers/footers, text in figures
#     for i in text_items:
#         # FIXME still removes some subsection headers due that they arent tagged as SECTION_HEADER
#         # each of the conditions has a drawback
#         #   char threshold removes some subsection tiles thus apply it not for text_items marked as SECTION_HEADER,
#         #   "BODY" includes also words in images,
#         # IDEA check intermediate markdown (Korzilius, Mohr) if subsection headers are rendered by \n\n <subsection header> \n or similarly
#         if (
#             i.content_layer.name == "BODY"
#             and len(i.text) < 50
#             and i.label.name != "SECTION_HEADER"
#         ):
#             text_items_to_drop.append(i)
#             text_items_to_drop_visualization.append([len(i.text), i.text])

#     ## drop selected text items
#     conv_file.document.delete_items(node_items=text_items_to_drop)

#     texts_cleaned = len(conv_file.document.texts)
#     print(f"Total texts after deletion: {texts_cleaned}")

#     return conv_file

def save_doc_to_jsonl(array:Iterable[Document], file_path:str)->None:
    """ Saving langchain for DocumentLoader to jsonl"""
    # taken from: https://github.com/langchain-ai/langchain/issues/3016
    with open(file_path, "w") as f: #, encoding="utf-8") as f:
        for d in array:
            f.write(d.json() + '\n')


def load_doc_from_jsonl(file_path:str)->Iterable[Document]:
    """ Loading Documents for DocumentLoader from jsonl"""
    # taken from: https://github.com/langchain-ai/langchain/issues/3016
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array
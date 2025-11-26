#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parse and clean text sources"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import re
import warnings

from docling_core.types.doc.document import TextItem
from docling.document_converter import ConversionResult


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


def remove_headers_footers(conv_file: ConversionResult) -> ConversionResult:
    ## remove headers and footers from the document
    total_texts = len(conv_file.document.texts)
    print(f"Total texts in document: {total_texts}")

    text_items = [x for x in conv_file.document.texts if isinstance(x, TextItem)]

    text_items_to_drop = []
    text_items_to_drop_visualization = []

    ## select text items to drop based on their number of chars, e.g. headers/footers, text in figures
    for i in text_items:
        # FIXME still removes some subsection headers due that they arent tagged as SECTION_HEADER
        # each of the conditions has a drawback
        #   char threshold removes some subsection tiles thus apply it not for text_items marked as SECTION_HEADER,
        #   "BODY" includes also words in images,
        # IDEA check intermediate markdown (Korzilius, Mohr) if subsection headers are rendered by \n\n <subsection header> \n or similarly
        if (
            i.content_layer.name == "BODY"
            and len(i.text) < 50
            and i.label.name != "SECTION_HEADER"
        ):
            text_items_to_drop.append(i)
            text_items_to_drop_visualization.append([len(i.text), i.text])

    ## drop selected text items
    conv_file.document.delete_items(node_items=text_items_to_drop)

    texts_cleaned = len(conv_file.document.texts)
    print(f"Total texts after deletion: {texts_cleaned}")

    return conv_file

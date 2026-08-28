#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parse and clean text sources"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

from pathlib import Path
import re
import warnings
from contextlib import suppress
from typing import List, Dict, Tuple, Optional, Union
import json

import nltk
from typing import Iterable
from docling_core.types import DoclingDocument
from docling_core.types.doc import CoordOrigin
from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, DocItem
from langchain_core.documents import Document
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, AcceleratorOptions, ApiVlmOptions, ResponseFormat, VlmPipelineOptions
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.document_converter import ConversionResult, DocumentConverter, PdfFormatOption, ImageFormatOption, PipelineOptions, InputFormat
from haystack.dataclasses import ByteStream

import requests
# from docling.datamodel.base_models import InputFormat
from docling.datamodel.settings import settings


class DocumentParser:
    """Class for parsing and cleaning documents."""

    def __init__(self):
        # OCR pipeline configs
        self.artifacts_path = Path("../docling_artifacts")
        self.artifacts_path.mkdir(exist_ok=True)

        # EASY-OCR and pipeline options
        ocr_options = EasyOcrOptions(
            lang=["fr", "de", "es", "it", "nl"],
            # lang=["en", "fr", "de", "es", "pt", "it", "pl", "cs", "nl", "da", "sv", "no", "hr", "ro", "bg" "sl", "sk", "lt", "et" ],
            download_enabled=True
        )
        pipeline_opts = PdfPipelineOptions(
            artifacts_path=self.artifacts_path,
            do_ocr=True,         # Required for text extraction
            do_table_structure=False,  # Disable table analysis as not needed, and not well working for EasyOCR
            generate_picture_images=False, 
            #allow_external_plugins=True,
            ocr_options=ocr_options
        )
        
        # PDF format options
        pdf_format_option = PdfFormatOption(
            pipeline_options=pipeline_opts,
            # reading_order="natural"
        )
        # init pdf converter
        self.ocr_converter = DocumentConverter(
            format_options={InputFormat.PDF: pdf_format_option}
        )
        
        # # for RAPID-OCR 
        # # NOTE: needs language detection beforehand and DWL of respective model, and issue of detecting text in 2-column pages eg. frontpage Koks 2022
        # # !uv add rapidocr_onnxruntime




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
            text_items_to_drop_visualization.append([len(i.text), i.text, i])

    ## drop selected text items
    conv_file.document.delete_items(node_items=text_items_to_drop)

    print(f"Total texts after deletion: {len(conv_file.document.texts)}")

    return conv_file, text_items_to_drop_visualization


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



### taken and adapted from https://github.com/brucenielson/BookSearchArchive/blob/e2d6c4145d7931648d5854ba29186cbec8150e87/docling_parser.py
### and related blog post: https://www.mindfiretechnology.com/blog/archive/finding-paragraphs-in-pdfs-using-ibm-s-docling/
def clean_text(p_str: str) -> str:
    p_str = str(p_str).strip()  # Convert text to a string and remove leading/trailing whitespace
    p_str = p_str.encode('utf-8').decode('utf-8')
    p_str = re.sub(r'\s+', ' ', p_str).strip()  # Replace multiple whitespace with single space
    p_str = re.sub(r"([.!?]) '", r"\1'", p_str)  # Remove the space between punctuation (.!?) and '
    p_str = re.sub(r'([.!?]) "', r'\1"', p_str)  # Remove the space between punctuation (.!?) and "
    p_str = re.sub(r'\s+\)', ')', p_str)  # Remove whitespace before a closing parenthesis
    p_str = re.sub(r'\s+]', ']', p_str)  # Remove whitespace before a closing square bracket
    p_str = re.sub(r'\s+}', '}', p_str)  # Remove whitespace before a closing curly brace
    p_str = re.sub(r'\s+,', ',', p_str)  # Remove whitespace before a comma
    p_str = re.sub(r'\(\s+', '(', p_str)  # Remove whitespace after an opening parenthesis
    p_str = re.sub(r'\[\s+', '[', p_str)  # Remove whitespace after an opening square bracket
    p_str = re.sub(r'\{\s+', '{', p_str)  # Remove whitespace after an opening curly brace
    p_str = re.sub(r'(?<=\s)\.([a-zA-Z])', r'\1',
                   p_str)  # Remove a period that follows a whitespace and comes before a letter
    p_str = re.sub(r'\s+\.', '.', p_str)  # Remove any whitespace before a period

    # Remove footnote numbers at end of a sentence. Check for a digit at the end and drop it
    # until there are no more digits or the sentence is now a valid end of a sentence.
    while p_str and p_str[-1].isdigit() and not is_sentence_end(p_str):
        p_str = p_str[:-1].strip()
    
    return p_str


##########################  MY FUNCS
def remove_figure_references(p_str: str) -> str:
    # remove potneital figure reference when they are colsed by bracketss, e.g. (A1), (B20)
    # this is done to avoid mismatches with road names
    p_str = re.sub(r"\s+\([A-Z][0-9]{1,}\)", "", p_str)
    return p_str

def is_reference_section(document_text: str) -> bool:
    # search for reference section
    pattern = re.compile(
        r"^(References|REFERENCES|Bibliography|BIBLIOGRAPHY)$", flags=re.MULTILINE
    )
    # re.MULTILINE in combination with "^" and case sensitive : find search words only when they are at beginning of a new line
    matches = re.findall(pattern, document_text)
    if matches:
        print(f"Reference section found!" )
        return True

def is_conclusions_section(document_text: str) -> bool:
    # search for disucssion/conclusion section
    pattern = re.compile(
        r"Discussion|discussion|Conclusions|conclusions", flags=re.MULTILINE
    )
    # re.MULTILINE in combination with "^" and case sensitive : find search words only when they are at beginning of a new line
    matches = re.findall(pattern, document_text)
    if matches:
        return True

def is_abstract(document_text: str) -> bool:
    # search for abstract section or "Abstract." mentioning in text-body-size (e.g .Koks2022, kettle2020)
    pattern = re.compile(r"^(ABSTRACT|Abstract|SUMMARY|FOREWORD|Summary|Zusammenfassung)$|Abstract.", flags=re.MULTILINE)
    # re.MULTILINE in combination with "^" and case sensitive : find search words only when they are at beginning of a new line
    matches = re.findall(pattern, document_text)
    if matches:
        return True

def is_not_running_text(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    # find figure or table text or figure/table text marked as group, e.g. axis-labels, table text, figure annotations etc.  (<= 3 words, parent=picture). 
    pattern = re.compile(r"(#/pictures/*|#/tables/*|#/groups/*)", flags=re.MULTILINE)
    matches = re.findall(pattern, text.parent.cref) #and (len(text.text.split()) <= 3)
    if matches:
        return True



def is_caption(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label._value_ == "caption"

###################



def is_sentence_end(text: str) -> bool:
    has_end_punctuation: bool = is_ends_with_punctuation(text)
    # Does it end with a closing bracket, quote, etc.?
    ends_with_bracket: bool = (text.endswith(")")
                               or text.endswith("]")
                               or text.endswith("}")
                               or text.endswith("\"")
                               or text.endswith("\'"))
    return (has_end_punctuation or
            (ends_with_bracket and is_ends_with_punctuation(text[0:-1])))


def combine_paragraphs(p1_str: str, p2_str: str):
    # If the paragraph ends without final punctuation, combine it with the next paragraph
    if is_sentence_end(p1_str):
        return p1_str + "\n" + p2_str
    else:
        return p1_str + " " + p2_str



def is_section_header(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    if text is None:
        return False
    return text.label._value_ == "section_header"


def is_page_footer(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label._value_ == "page_footer"


def is_page_header(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label._value_ == "page_header"


def is_footnote(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label._value_ == "footnote"


def is_list_item(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label._value_ == "list_item"


def is_text_break(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return is_page_header(text) or is_section_header(text) or is_footnote(text)


def is_page_not_text(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label._value_ not in ["text", "list_item", "formula"]


def is_page_text(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return not is_page_not_text(text)


def is_ends_with_punctuation(text: str) -> bool:
    return text.endswith(".") or text.endswith("?") or text.endswith("!")


def is_too_short(doc_item: DocItem, threshold: int = 2) -> bool:
    return doc_item.label._value_ == "text" and len(doc_item.text) <= threshold


def is_bottom_note(text: DocItem, near_bottom: bool = False) -> bool:
    # If it is specifically digits followed by a period, followed by a space, and it is
    # a section header or a list item, then it is NOT a bottom note
    if bool(re.match(r"^\d+\.\s", text.text)) and (is_section_header(text) or is_list_item(text)):
        return False
    # If it's digits followed by a letter without a space then it's a bottom note
    if bool(re.match(r"^\d+[A-Za-z]", text.text)):
        return True

    if text is None or not is_page_text(text):
        return False
    # Check for · at the beginning of the line. This is often how OCR represents footnote number.
    if text.text.startswith("·") and not text.text.startswith("· "):
        return True

    if re.match(r"^\d", text.text):
        # If the first digit is zero, it can't be a footnote because that should never happen.
        if text.text.startswith("0"):
            return False
        if near_bottom:
            # Check if this is three digits with the third digit being a 1 followed by a space
            # This is usually where the last 1 was supposed to be an 'I'.
            return re.match(r"^\d{1,2}1 ", text.text) or not is_list_item(text)

    return False


def is_near_bottom(doc_item: DocItem, same_page_items: [DocItem], threshold: float = 0.5) -> bool:
    """
    Determine if a DocItem is near the bottom of its page.

    Parameters:
    - doc_item: The DocItem object containing provenance data with 'bbox'.
    - doc: The DoclingDocument containing all DocItems.
    - threshold: Distance in points from the bottom to consider as 'near the bottom'.

    Returns:
    - True if the DocItem is within the threshold from the bottom, False otherwise.
    """
    # Check if the DocItem has provenance data with a bounding box
    if hasattr(doc_item.prov[0], 'bbox'):
        bbox = doc_item.prov[0].bbox
    else:
        return False  # No bounding box available

    # Extract the coordinate origin and bounding box coordinates
    coord_origin = bbox.coord_origin
    x0, y0, x1, y1 = bbox.l, bbox.b, bbox.r, bbox.t

    # Find the maximum y1 value on the page
    page_top: float = max(item.prov[0].bbox.t for item in same_page_items if hasattr(item.prov[0], 'bbox'))
    # Find the min y1 value on the page
    page_bottom: float = min(item.prov[0].bbox.b for item in same_page_items if hasattr(item.prov[0], 'bbox'))
    page_size: float = page_top - page_bottom
    # Threshold is page_bottom + (size of page * threshold amount) (i.e. % of page to be considered the 'bottom')
    bottom_threshold: float = page_bottom + (page_size * threshold)

    if coord_origin == CoordOrigin.BOTTOMLEFT:
        # In this system, y1 is the distance from the top of the paragraph to the bottom of the page
        return y1 <= bottom_threshold
    elif coord_origin == CoordOrigin.TOPLEFT:
        # In this system, y1 is the distance from the top of the paragraph to the top of the page
        return y1 >= bottom_threshold
    else:
        raise ValueError("Unknown coordinate origin.")



def is_text_item(item: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return not (is_section_header(item)
                or is_page_footer(item)
                or is_page_header(item)
                # or is_reference_section(item)
            )



def get_next_text(texts: List[Union[SectionHeaderItem, ListItem, TextItem]], i: int) \
        -> Optional[Union[ListItem, TextItem]]:
    # Seek through the list of texts to find the next text item using is_text_item
    # Should return None if no more text items are found
    for j in range(i + 1, len(texts)):
        if j < len(texts) and is_text_item(texts[j]):  # skips page headers/footers
            return texts[j]
    return None


def is_roman_numeral(s: str) -> bool:
    roman_numeral_pattern = r'(?i)^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(roman_numeral_pattern, s.strip()))

def should_skip_element(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return any([
        is_page_footer(text),
        is_page_header(text),
        is_roman_numeral(text.text),
        is_not_running_text(text),
        is_footnote(text),
        is_caption(text)
    ])


def get_processed_texts(doc: DoclingDocument) -> List[DocItem]:
    """
    Processes the document's text items page by page, separating regular content from notes
    (footnotes and bottom notes), and returns a list of DocItems with notes at the end.
    """
    regular_texts: List[DocItem] = []
    notes: List[DocItem] = []
    processed_pages: set[int] = set()  # Keep track of processed pages
    reached_bottom_notes: bool = False
    same_page_items: List[DocItem] = []
    near_bottom: bool = False
    mislabeled: List[DocItem] = []

    for text_item in doc.texts:
        page_number = text_item.prov[0].page_no

        if page_number not in processed_pages:
            # On new page, so get all items on the current page
            same_page_items = [
                item for item in doc.texts if item.prov[0].page_no == page_number
            ]
            processed_pages.add(page_number)  # Mark the page as processed
            reached_bottom_notes = False

        if not reached_bottom_notes:
            near_bottom = is_near_bottom(text_item, same_page_items, threshold=0.5)

        if is_too_short(text_item):
            continue
        elif reached_bottom_notes or is_footnote(text_item):
            # NOTE: # workaround to keep paragraphs in correct section 
            # issue: some text paragraphs are removed as they get the wrong section_name, this bc they are getting the label reached_bottom_notes=True and thenjust added later to the outp of get_processed_text() 
            if len(text_item.text) > 3:
                regular_texts.append(text_item)
            notes.append(text_item)
        elif is_bottom_note(text_item, near_bottom=near_bottom):
            notes.append(text_item)
            reached_bottom_notes = True
        else:
            regular_texts.append(text_item)

        # Check if the DocItem is a SectionHeaderItem. If so, turn it into a TextItem.
        if is_section_header(text_item):
            mislabeled.append(text_item)

    # new_texts = regular_texts + notes
    new_texts = regular_texts # NOTE test without notes, this text that is already added by " workaround to keep paragraphs in correct section "
   

    # # is reference or conlcusion
    # if dc.is_conclusions_section(section_name):
    #     print("Discussion / Conclusions section found. Stopping further processing of document.")
    #     continue 
    # if dc.is_reference_section(section_name):
    #     print("Reference section found. Stopping further processing of document.")
    #     continue                    

    # # skip paragraph belonging to Abstract/Summary
    # # keep "overview" section as might be more like introduction with detailed info, recognized also "Abstract." in text-body fontsize
    # if (dc.is_abstract(section_name)) or (dc.is_abstract(text.text)):
    #     print("Abstract/Summary section found. Continue with next text piece.")
    #     continue

    return new_texts

    # return regular_texts + notes



def add_paragraph(
    text: str,               
    # para_num: int, section: str, page: Optional[int], 
    docs: List[ByteStream], 
    # meta: List[Dict]
):
    docs.append(ByteStream(text.encode('utf-8')))
    # meta.append({
    #     **meta_data,
    #     # "paragraph_#": str(para_num),
    #     "section_name": section,
    #     "page_#": str(page)
    # })



##########################  HYPHEN CLENAING SOLO  (keep solo or inc. in DoclingParser class)

## document-wise cleaning 

# Module-level caches (private)
_words_list = None
_lemmatizer = None
_stemmer = None


# ## load nltk libs for handling hyphens
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def get_words_list():
    """Lazily load and cache the NLTK english words list."""
    global _words_list
    if _words_list is None:
        nltk.download('words')
        _words_list = set(nltk.corpus.words.words())
    return _words_list


def get_lemmatizer():
    """Lazily load and cache the WordNetLemmatizer."""
    global _lemmatizer
    if _lemmatizer is None:
        from nltk.stem import WordNetLemmatizer
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def get_stemmer():
    """Lazily load and cache the PorterStemmer."""
    global _stemmer
    if _stemmer is None:
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
    return _stemmer


def is_valid_word(word):
    """
    Check if a word is valid by comparing it directly and via stemming/lemmatization.
    In detail, it checks if the given word, its stem, or its lemma is inlcuded in the word list downloaded from nltk or the customized list of suffixes.

    Returns True (or the valid modified word) if the word is found,
    otherwise returns False.
    """
    words_list = get_words_list()
    stemmer = get_stemmer()
    lemmatizer = get_lemmatizer()

    stem = stemmer.stem(word)
    if word.lower() in words_list or word in words_list:
        return True
    elif stem in words_list or stem.lower() in words_list:
        return True

    # Check all lemmatizations of the word
    for pos in ['n', 'v', 'a', 'r', 's']:
        lemma = lemmatizer.lemmatize(word, pos=pos)
        if lemma in words_list:
            return True

    # Check for custom lemmatizations
    # noinspection SpellCheckingInspection
    suffixes = {
        "ability": "able",  # testability -> testable
        "ibility": "ible",  # possibility -> possible
        "iness": "y",  # happiness -> happy
        "ity": "e",  # creativity -> create
        "tion": "e",  # creation -> create
        "able": "",  # testable -> test
        "ible": "",  # possible -> poss
        "ing": "",  # running -> run
        "ed": "",  # tested -> test
        "s": ""  # tests -> test
    }
    for suffix, replacement in suffixes.items():
        if word.endswith(suffix):
            stripped_word = word[:-len(suffix)] + replacement
            # Recursively check the modified word; if valid, return the modified form.
            result = is_valid_word(stripped_word)
            if result:
                return result

    return False


def combine_hyphenated_words(p_str):
    """
    Combine hyphenated words if the parts together form a valid word.
    Otherwise, preserve the hyphen (assuming it connects two valid words).
    """

    def replace_dash(match):
        word1, word2 = match.group(1), match.group(2)
        combined = word1.strip() + word2.strip()

        # If there is a space after the hyphen and the combined word is valid,
        # assume the hyphen was splitting a single word.
        # if word2.startswith(" ") and is_valid_word(combined):  # org
        # NOTE: test if it not merges two proper nouns, eg. Sachsen-Anhalt, Elbe-Havel
        if word2.startswith(" ") and is_valid_word(combined) and word2.strip()[0].islower():
            return combined
        # If both parts are valid words on their own, keep them hyphenated.
        elif is_valid_word(word1.strip()) and is_valid_word(word2.strip()):
            return word1.strip() + '-' + word2.strip()
        # Otherwise, if the combined word is valid, return it.
        elif is_valid_word(combined):
            return combined
        # If the combined word starts with a capital letter (likely a proper noun)
        # and the second part isn’t valid on its own, combine them.
        elif combined[0].isupper() and not word2.strip()[0].isupper() and not is_valid_word(word2.strip()):
            return combined

        # Default: assume the hyphen is meant to connect two words.
        return word1.strip() + '-' + word2.strip()

    # Replace any soft hyphen characters with a regular dash.
    p_str = p_str.replace("­", "-")
    # Look for hyphens between word parts (with or without an extra space)
    p_str = re.sub(r'(\w+)-(\s?\w+)', replace_dash, p_str)

    return p_str

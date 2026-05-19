#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multilingual Translator from any EU-language -> english"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"


import gc
import os
from annotated_types import doc
import langdetect

from huggingface_hub import login
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)
import torch

from src.settings import settings as s


# Multilingual Translator 

# should be enough for sentence wise translation of each chunk (limiation: no context-related translation based on previous text)
# taken from: https://thepythoncode.com/article/machine-translation-using-huggingface-transformers-in-python
def init_helsinki_nlp(src_language, dst_language) -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Given the source and destination languages, returns the appropriate model
    See the language codes here: https://developers.google.com/admin-sdk/directory/v1/languages
    For the 3-character language codes, you can google for the code!
    """

    # print("Login with HF TOKEN ...")    
    # login(token=os.environ["HUGGINGFACE_TOKEN"])

    gc.collect()
    torch.cuda.empty_cache() 
    torch.no_grad()    

    # construct our model name
    model_name = f"Helsinki-NLP/opus-mt-{src_language}-{dst_language}"

    base_dir =  s.HF_HOME_DIR 
    model_dir = base_dir 
    print(model_dir)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model and Tokenizer initialization
    if not os.path.exists(model_dir):
        print("Model directory not found. Downloading model...")
        os.makedirs(model_dir, exist_ok=True)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype="auto",
            quantization_config=bnb_config,
        )
        model.save_pretrained(model_dir)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=True,
        )
        tokenizer.save_pretrained(model_dir)

    else:
        print(f"Using locally saved model from {model_dir}/{model_name}")

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=model_dir,
            local_files_only=False,  # set to False to enable downloading if model is not found locally, even when it exists
            # tp_plan="auto" # set tensor parallel model (ie. splits model on multiple GPU)
            dtype="auto",
            quantization_config=bnb_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            cache_dir=model_dir,
            backend="sentencepiece"
        )

    ## reduce further memory usage
    model = model.to(device)
    model.use_checkpointing = True
    torch.cuda.empty_cache()

    return model, tokenizer



def translate_2_english(src_language_doc: str, doc: list[str] | str) -> list[str] | str:
    """Translate Docling Document of any supported languages to English using Helsinki NLP models"""

    dst_language = "en"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load appropriate Helsinki model and tokenizer
    model, tokenizer = init_helsinki_nlp(src_language_doc, dst_language)

    # TODO condense try except blocks when handling string input
    # ISSUE: when doc is list[doc] then translation per chunk is needed, but when doc is single string then direct translation
    
    if isinstance(doc, list):  # doc == Docling object or ConversionResult

        for j, chunk in enumerate(doc):

            try:
                src_text = chunk.page_content
            except AttributeError as e:
                src_text = chunk.text

            if src_text.strip() == "": 
                continue

            # detect language type for each chunk if text is not empty or too short
            try:
                src_language = langdetect.detect(src_text)
            except langdetect.lang_detect_exception.LangDetectException as e:
                print(f"Language detection failed for chunk {j} with text: {src_text[:30]}... Skipping translation for this chunk.")
                continue
            # print(f"Detected language for chunk {j}: {src_language}")

            supported_languages = ["fr", "de", "es", "it", "nl"]  # TODO make as global var in config file

            if (src_language == dst_language) or (src_language not in supported_languages):
                # print(f"Source and destination language in chunk {j} are identical or unsupported. No translation needed.")
                # write back to document  # TODO condense try except blocks when handling string input
                try:
                    doc[j].page_content =  src_text
                except ValueError:
                    doc[j].text =  src_text

            # tokenize the input text and move to the appropriate device
            inputs = tokenizer.encode(src_text, return_tensors="pt", max_length=2048, truncation=True)
            inputs = inputs.to(device)

            # generate the translation output using greedy search, 
            # TODO recheck difference to greedy search
            beam_outputs = model.generate(inputs, num_beams=3)

            # decode the output and ignore special tokens
            dst_text = tokenizer.decode(beam_outputs[0], skip_special_tokens=True)

            # # decode the output and ignore special tokens
            # print(" Source text: ", src_text)
            # print(" Translated text: ", tokenizer.decode(beam_outputs[0], skip_special_tokens=True))
        
            # write back to document
            try:
                doc[j].page_content =  dst_text
            except ValueError as e:
                doc[j].text =  dst_text


    if isinstance(doc, str):
        print("Input document is a string (not DoclingObject). Wrapping it in a list for processing.")
        print("Continue with translation of text string")
        
        src_text = doc

        # detect language type for each chunk if text is not empty or too short
        src_language = langdetect.detect(src_text)

        try:    
            # tokenize the input text and move to the appropriate device
            inputs = tokenizer.encode(src_text, return_tensors="pt", max_length=2048, truncation=True)
            inputs = inputs.to(device)
    
            # generate the translation output using beam search
            beam_outputs = model.generate(inputs, num_beams=3)
    
            # print("Source text: ", src_text)
            # print("Translated text: ", tokenizer.decode(beam_outputs[0], skip_special_tokens=True))
    
            # decode the output and ignore special tokens
            dst_text = tokenizer.decode(beam_outputs[0], skip_special_tokens=True)
    
            # write back to document
            doc = dst_text
            
        except langdetect.lang_detect_exception.LangDetectException as e:
            print(f"Language detection failed for chunk {j} with text: {src_text[:30]}... Skipping translation for this chunk.")
            
            doc = src_text
        
            supported_languages = ["fr", "de", "es", "it", "nl"]  # TODO make as global var in config file
            if (src_language == dst_language) or (src_language not in supported_languages):
                print("Source and destination language for text are identical or unsupported. Skipping translation.")
                # write back to document
                doc = src_text


  
    # cleaning up memory after translation
    gc.collect()
    torch.no_grad()

    return doc
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multilingual Translator from any EU-language -> english"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"


import os
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
    # construct our model name
    model_name = f"Helsinki-NLP/opus-mt-{src_language}-{dst_language}"


    print("Login with HF TOKEN ...")    
    login(token=os.environ["HUGGINGFACE_TOKEN"])

    base_dir =  s.HF_HOME_DIR 
    model_dir = base_dir # is already .._mirror/hub/ 
    print(model_dir)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = transformers.infer_device()
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
        print(f"Using locally saved model from {model_dir}")

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=model_dir,
            local_files_only=True,  # tp_plan="auto" # set tensor parallel model (ie. splits model on multiple GPU)
            dtype="auto",
            quantization_config=bnb_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            cache_dir=model_dir,
        )
  
    ## reduce further memory usage
    model = model.to(device)
    model.use_checkpointing = True
    torch.cuda.empty_cache()

    return model, tokenizer



def translate_2_english(src_text: str) -> str:
    """Translate any text in supported languages to english using a Helsinki NLP model"""

    dst_language = "en"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # detect language type and availability 
    src_language = langdetect.detect(src_text)
    print(f"Detected language: {src_language}")

    supported_languages = ["en", "fr", "de", "es", "it", "nl"]
    if src_language not in supported_languages:
        raise ValueError(f"Unsupported source language: {src_language}")

    # load appropriate Helsinki model and tokenizer
    model, tokenizer = init_helsinki_nlp(src_language, dst_language)

    # tokenize the input text and move to the appropriate device
    inputs = tokenizer.encode(src_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)

    # generate the translation output using greedy search
    greedy_outputs = model.generate(inputs)

    # decode the output and ignore special tokens
    print(tokenizer.decode(greedy_outputs[0], skip_special_tokens=True))

    # generate the translation output using beam search
    beam_outputs = model.generate(inputs, num_beams=3)

    # decode the output and ignore special tokens
    dst_text = tokenizer.decode(beam_outputs[0], skip_special_tokens=True)

    return dst_text



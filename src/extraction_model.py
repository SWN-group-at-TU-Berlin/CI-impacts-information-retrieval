#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data mining model (decoder, tokenizer) for extracting impacts on infrastructure"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"



import os

from jinja2 import Environment, FileSystemLoader
from huggingface_hub import login
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DynamicCache,
)

import torch

from src.settings import settings as s
import src.utils as u




def load_prompt_template(
    template_path: str = "./prompt_templates",
    template_filename: str = "ci_loc_direct_impacts.txt"
):
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template(template_filename)

    return template



class DecoderModel:

    def __init__(self, model_name: str ="meta-llama/Llama-2-7b-chat-hf",):
        
        try: 
            login(token=os.getenv("HUGGINGFACE_TOKEN"))   # notebook_login
        except:
            login(token=os.environ.get("HUGGINGFACE_TOKEN"))  # former HF_TOKEN

        base_dir =  s.HF_HOME_DIR   # use default dir in .cache/
        model_dir = base_dir # / f"models--{model_name.replace("/", "--")}"  # is already .._mirror/hub/
        
        print(model_dir)

        # quantization config
        # Load model with 4-bit quantization if applicable (use 4-bit integer instead of 32b floats) --> reduce the required VRAM for model application
        # see, https://huggingface.co/docs/transformers/quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.pipeline, self.tokenizer, self.past_key_values = self.initialize_model(
            model_name, model_dir, bnb_config
        )
        

    def initialize_model(self, model_name: str, model_dir: str = None, bnb_config=None, max_new_tokens: int = 1024):
        
        # use flash-attn when GPU type supports it (e.g., A100, not support:tesla P100)
        if u.supports_flash_attention(0):  # check only for first GPU
            print("Using flash attention")
            flash_attn_config = "flash_attention_2"
        else:
            print("Flash attention not supported for this GPU")
            flash_attn_config = None

        # Model and Tokenizer initialization
        if not os.path.exists(model_dir):
            print("Model directory not found. Downloading model...")
            os.makedirs(model_dir, exist_ok=True)

            device = transformers.infer_device()
            print(f"Using device: {device}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype="auto", # None ,# test for CU12.6, torch.29.1 #"auto",
                # max_seq_length=2048,
                attn_implementation=flash_attn_config,
                quantization_config=bnb_config,
                # max_memory={0: "2GB", 1: "10GB"},  # distribute memory across GPUs
            )
            model.save_pretrained(model_dir)
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            tokenizer.save_pretrained(model_dir)

            print("Downloaded model and tokenizer")

        else:
            print(f"Using locally saved model from {model_dir}")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=model_dir,
                local_files_only=True,  # tp_plan="auto" # set tensor parallel model (ie. splits model on multiple GPU)
                # max_seq_length=2048,
                dtype="auto", # None ,# test for CU12.6, torch.29.1 #"auto",
                attn_implementation=flash_attn_config,
                quantization_config=bnb_config,
                # tp_plan="auto",  # automatically use a tensor parallelism plan based on predefined configuration of the model (i.e. partition model on both GPUs)
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                cache_dir=model_dir,  # use fast Rust-based tokenizer, when possible
            )

        ## reduce memory usage
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.use_checkpointing = True
        
        torch.cuda.empty_cache()
        torch.no_grad()

        ## Caching
        # set iterative generation to avoid recomputing entire prompt
        past_key_values = DynamicCache(config=model.config)



        # Pipeline setup for question answering
        pipeline = transformers.pipeline(  # load model locally from wsl .cache\
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens, # high max token otherwise output is truncated
            device_map="auto",
        )
        return pipeline, tokenizer, past_key_values


    def generate_response(
        self, question: str, context: list, 
        prompt_template: load_prompt_template, #chunk_id: int
        top_k: int = None,
        top_p: float = None,
        temperature: float = 0.2,
        max_new_tokens: int = 1024
    ):
        rendered_prompt = prompt_template.render(
            context=context,  # includes also df_ci_geo info
            question=question,
        )

        # print(f"Generating response for chunk_id: {chunk_id} ...")

        sequences = self.pipeline(
            rendered_prompt,  # jinja template
            max_new_tokens=max_new_tokens, # use default to not truncate the LLM response
            do_sample=True,
            num_beams=1,  # select token based on probability distribution over entire model’s vocabulary
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            # num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            past_key_values=self.past_key_values,
            return_full_text=False,  # allow bullet point answers
        )
        # Extracting and returning the generated text
        return sequences

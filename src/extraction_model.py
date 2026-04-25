#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data mining model (decoder, tokenizer) for extracting impacts on infrastructure"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"



import os
import copy

from jinja2 import Environment, FileSystemLoader
from huggingface_hub import login
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
    template_filename: str = None,
):
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template(template_filename)

    return template



class DecoderModelCaching:

    def __init__(self, model_name: str, static_prompt: load_prompt_template):
        
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

        self.model, self.tokenizer, self.prompt_cache = self.initialize_model(
            model_name, model_dir, bnb_config,
            static_prompt=static_prompt,
        )
        

    def initialize_model(
        self, model_name: str, model_dir: str = None, 
        bnb_config=None, 
        static_prompt=None, 
        max_new_tokens: int = 2048 # 4096
    ):
            
        # use flash-attn when GPU type supports it (e.g., A100, not support:tesla P100)
        flash_attn_config = None
        if u.supports_flash_attention(0):  # check only for first GPU
            print("Using flash attention")
            flash_attn_config = "flash_attention_2"

        # Model and Tokenizer initialization
        if not os.path.exists(model_dir):
            print("Model directory not found. Downloading model...")
            os.makedirs(model_dir, exist_ok=True)

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
            print(f"Using locally saved model from {model_dir}/{model_name}")

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
        # past_key_values = DynamicCache(config=model.config)
        print("Using iterative caching of prompt key values to avoid recomputing entire prompt for each generation step.")
        # print("Using offloading currently") # to CPU for prompt cache to reduce GPU memory usage")
        prompt_cache =  DynamicCache(config=model.config) #, offloading=True) 
        static_prompt = static_prompt.render()
        inputs_initial_prompt = tokenizer(static_prompt, return_tensors="pt").to(model.device.type)
        # This is the common prompt cached, we need to run forward without grad to be able to copy
        with torch.no_grad():
            prompt_cache = model(**inputs_initial_prompt, past_key_values=prompt_cache).past_key_values

        return model, tokenizer, prompt_cache


    def generate_response(
        self, 
        question: str, context: list, 
        static_prompt: load_prompt_template, 
        dynamic_prompt: load_prompt_template, 
        num_beams: int = 1,
        top_k: int = None,
        top_p: float = None,
        temperature: float = 0.2,
        max_new_tokens: int = 1024
    ):
        static_prompt = static_prompt.render()
        dynamic_prompt = dynamic_prompt.render(
            context=context,  # includes also df_ci_geo info
            question=question,
        )
        
        new_inputs = self.tokenizer(static_prompt + dynamic_prompt, return_tensors="pt").to(self.model.device.type)

        # print("--> Using currently offloading and only shallow copy of pk_values")
        # past_key_values = copy.copy(self.prompt_cache)  
        # print("--> Not using offloading, but deep copy of pk_values")
        past_key_values = copy.deepcopy(self.prompt_cache)  # Needed to copy past KV values
        # # FIXME - potential issues as pk_v is not copied as completely indenpendent object (set offloading=True) 

        outputs = self.model.generate(
            **new_inputs, 
            temperature=temperature,
            past_key_values=past_key_values, 
            do_sample=True,  # more creative output - NOTE: avoid duplications, better results than when False (greedy decoding)
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id
            )

        # Extracting and returning the generated text
        sequences = self.tokenizer.batch_decode(outputs)[0]
        try:
            sequences_cleaned = sequences.split("ANSWER:")[1]       
        except:
            sequences_cleaned = sequences.split("The final answer is:")[1]       

        return sequences_cleaned



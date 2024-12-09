
"""This file is to create a new tokenizer and model with added tokens for graph input."""
from transformers import PreTrainedModel, AutoConfig, AutoTokenizer
import torch
import json
import os
import numpy as np
from gated_mem_Llama import GMLlamaForCausalLM
PAD_TOKEN = '[PAD]'
import argparse

def main():
    tokenizer = AutoTokenizer.from_pretrained("./Meta-Llama-3.1-8B-Instruct", local_files_only=True)
    model_config = AutoConfig.from_pretrained("./Meta-Llama-3.1-8B-Instruct", local_files_only=True)
    model = GMLlamaForCausalLM.from_pretrained("./Meta-Llama-3.1-8B-Instruct", config=model_config, local_files_only=True, low_cpu_mem_usage=False, _fast_init=False)
    model.save_pretrained("./newmodel")
    tokenizer.save_pretrained("./newmodel")


if __name__ == '__main__':
    main()

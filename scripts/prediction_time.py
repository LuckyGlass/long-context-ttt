"""
Used to evaluate running time.
Settings:
1. Baseline: pretrained long-context model (Llama-3.1-8B-Instruct), full ICL.
2. Quantized: pretrained long-context model (Llama-3.1-8B-Instruct), 4-bit, full ICL.
3. TTT: 5-epoch, truncated ICL.
4. SFT+TTT: 5-epoch, truncated ICL, expected to be the same as TTT.
"""
import json
import math
import numpy as np
import os
import time
import torch
import tqdm
from glob import glob
from transformers import PreTrainedTokenizer, TrainingArguments
from long_ttt.model import load_tokenizer, load_model
from long_ttt.train import train
from long_ttt.ttt_args import (
    ModelArguments,
    CustomTrainingArguments,
    DataTrainingArguments,
    parse_args
)
from long_ttt.utils import printGPU
from long_ttt.context_dataset import ContextDataset
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestArguments:
    setting: str = field(default='unk')
    haystack_path: str = field(default='unk')
    output_path: Optional[str] = field(default=None)


def generate_sample(
    tokenizer, 
    context, 
    context_length,
    needle_depth,
    needle="\n\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n", 
    prompt='\n\nWhat is the best thing to do in San Francisco?\nAnswer:',
    zh=False,
):
    """ It's modified for TTT use.
    Args:
        tokenizer:
    Returns:
        - inputs, the full input string (chat template applied).
        - context, the context (the context documents and the needle).
        - needle.
    """
    if zh:
        num_words = len(context)
    else:
        num_words = len(context.split())
    if context_length > num_words:
        context = context * math.ceil(context_length / num_words)

    if zh:
        description = "以下上下文中隐藏着重要信息。找到并记住这些信息。我会问你关于其中重要信息的问题。\n"
    else:
        description = "There is an important infomation hidden in the following context. Find the information and memorize it. I will quiz you about the important information there.\n"

    description_input_ids = tokenizer.encode(description, add_special_tokens=False)
    needle_input_ids = tokenizer.encode(needle, add_special_tokens=False)
    prompt_input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    description_length = len(description_input_ids)
    needle_length = len(needle_input_ids)
    prompt_length = len(prompt_input_ids)

    # must leave room for information and prompt
    minimum_pos = description_length
    maximum_pos = context_length - prompt_length - needle_length - 1
    if minimum_pos > context_length or maximum_pos < 0:
        raise ValueError(f"The length {context_length} is too small. Please increase interval!")

    needle_pos = minimum_pos + round((maximum_pos - minimum_pos) * needle_depth / 100)
    
    context_input_ids = tokenizer.encode(context, max_length=context_length - description_length - needle_length - prompt_length, truncation=True, add_special_tokens=False)

    input_ids = sum([description_input_ids, context_input_ids[:needle_pos], needle_input_ids, context_input_ids[needle_pos:], prompt_input_ids], [])
    context_ids = sum([context_input_ids[:needle_pos], needle_input_ids, context_input_ids[needle_pos:]], [])
    inputs = tokenizer.decode(input_ids)
    context_return = tokenizer.decode(context_ids)

    return inputs, context_return, needle


def single_test(model_name_or_path, tokenizer, length, setting, training_args, context, **kwargs):
    model_max_length = kwargs['model_max_length']
    model = load_model(model_name_or_path, tokenizer, **kwargs)
    input_str, context_str, _ = generate_sample(tokenizer, context, length, 50)
    input_ids = tokenizer(input_str, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
    gputime_1 = torch.cuda.Event(enable_timing=True)
    gputime_2 = torch.cuda.Event(enable_timing=True)
    if setting in ['Baseline', 'Quantized']:
        # Start the timer
        systime_1 = time.time()
        gputime_1.record()
        # Eval
        model.eval()
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
                max_new_tokens=10,
                use_cache=False,
            )
        # Stop the timer
        systime_2 = time.time()
        gputime_2.record()
        assert output_ids.shape[-1] == input_ids.shape[-1] + 10
    elif setting in ['TTT', 'SFT+TTT']:
        # Start the timer
        systime_1 = time.time()
        gputime_1.record()
        # Eval
        dataset = ContextDataset(context_str, tokenizer, **kwargs)
        model = train(dataset, tokenizer, training_args, model=model, **kwargs)[0]
        model.eval()
        if input_ids.shape[-1] > model_max_length:
            input_ids = torch.concat([input_ids[:, :model_max_length//2], input_ids[:, -model_max_length//2:]], dim=-1)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
                max_new_tokens=10,
                use_cache=False,
            )
        # Stop the timer
        systime_2 = time.time()
        gputime_2.record()
        assert output_ids.shape[-1] == input_ids.shape[-1] + 10
        del dataset
    torch.cuda.synchronize()
    systime_per_token = (systime_2 - systime_1) / 10
    gputime_per_token = (gputime_1.elapsed_time(gputime_2) / 1000) / 10
    del model, input_ids, output_ids
    return systime_per_token, gputime_per_token


def test(setting: str, output_path: str, tokenizer_name_or_path: str, model_name_or_path: str, haystack_path: str, training_args: TrainingArguments, **kwargs):
    tokenizer = load_tokenizer(tokenizer_name_or_path)
    context = ""
    file_names = [file for file in glob(f"{haystack_path}/*.txt")]
    num_tokens = 0
    for file in file_names:
        with open(file, 'r') as f:
            this_file_context = f.read()
            num_tokens += len(tokenizer.encode(this_file_context, add_special_tokens=False))
            context += this_file_context
            if num_tokens > 100000:
                break
    if os.path.exists(output_path):
        os.remove(output_path)
    test_lengths = [i * 10000 for i in range(1, 11)]
    for length in tqdm.tqdm(test_lengths, desc='Length'):
        systime = []
        gputime = []
        for _ in range(1):
            torch.cuda.empty_cache()
            printGPU()
            a, b = single_test(model_name_or_path, tokenizer, length, setting, training_args, context, **kwargs)
            systime.append(a)
            gputime.append(b)
        with open(output_path, 'a') as f:
            json.dump({'length': length, 'systime': np.mean(systime), 'gputime': np.mean(gputime)}, f)
            f.write('\n')


def main():
    args, training_args = parse_args(((TestArguments, ModelArguments, CustomTrainingArguments, DataTrainingArguments), TrainingArguments), no_dict=(TrainingArguments,))
    test(training_args=training_args, **args)


if __name__ == '__main__':
    main()

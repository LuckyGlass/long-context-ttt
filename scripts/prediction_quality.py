#!/usr/bin/env python
# coding=utf-8

import json
import tqdm
import torch
from transformers import TrainingArguments, HfArgumentParser, AutoTokenizer
from dataclasses import dataclass, field
import logging
from long_ttt.train import train
from long_ttt.ttt_args import (
    ModelArguments,
    CustomTrainingArguments,
    DataTrainingArguments,
    GlobalTestArguments,
    parse_args
)
from long_ttt.context_dataset import ContextDataset, apply_qa_template
from long_ttt.utils import get_average_attention, printGPU
from long_ttt.model import load_tokenizer
from typing import Optional
import os


@dataclass
class TestArguments(GlobalTestArguments):
    output_file: Optional[str] = field(default=None)


def prediction(training_args: TrainingArguments, args: dict, output_file: str, input_file: str="", overwrite: bool=True, config: Optional[dict]=None, **kwargs):
    # Resume from checkpoint
    results = [] if config is None else [config]
    if not overwrite and os.path.exists(output_file):
        logging.info(f"Detect existing output file {output_file}. Resume from the checkpoint.")
        with open(output_file, 'r') as f:
            results = json.load(f)
        logging.info(f"Load the results of {len(results)} samples.")
    # Pre-process
    model_max_length = args['model_max_length']
    debug_size = args.pop('debug_size')
    with open(input_file, "r") as f:
        samples = json.load(f)
        if debug_size is not None:
            samples = samples[:debug_size]
    
    for sample_id, sample in enumerate(tqdm.tqdm(samples, desc="Prediction")):
        if sample_id < len(results) - 1:
            continue
        torch.cuda.empty_cache()
        printGPU(f"Before training")
        tokenizer = AutoTokenizer.from_pretrained(args['model_name_or_path'])
        tokenizer.pad_token = tokenizer.eos_token
        context_dataset = ContextDataset(sample['input'], tokenizer, sample['title'], **args)
        model = train(context_dataset, tokenizer, training_args, **args)[0]
        model.eval()
        torch.cuda.empty_cache()
        printGPU(f"Eval with {len(sample['qa_pairs'])} samples")
        with torch.no_grad():
            for i, qa_pair in enumerate(tqdm.tqdm(sample['qa_pairs'], desc="QA")):
                summaries = qa_pair['summaries']
                answers = qa_pair['answers']
                prompts = [
                    "Please sort the given events in the order of their appearance in the following long texts, from first to last.",
                    sample['input'],
                    "Please sort the given events in the order of their appearance in the long texts, from first to last. The given events are:",
                ]
                prompts += [f"[{i + 1}]: {summaries[i]}" for i in range(len(summaries))]
                prompts += ["For example, a valid answer is [2] < [3] < [1] < [4] < [5]."]
                messages = [
                    {'role': 'system', 'content': "You are a helpful assistant."},
                    {'role': 'user', 'content': '\n'.join(prompts)},
                ]
                input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors='pt')
                if input_ids.shape[-1] > model_max_length:
                    input_ids = torch.concat((input_ids[:, :model_max_length//2], input_ids[:, -model_max_length//2:]), dim=-1)
                attention_mask = torch.ones_like(input_ids)
                terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=50,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=terminators,
                    do_sample=False,
                    temperature=1.0,
                    num_beams=1,
                    repetition_penalty=1.1,
                    return_dict_in_generate=True,
                )
                output_ids = outputs.sequences[0]
                qa_pair['pred'] = tokenizer.decode(output_ids[input_ids.shape[-1]:], skip_special_tokens=True)
        results.append(sample)
        del model, tokenizer
        with open(output_file, "w+") as f:
            json.dump(results, f, indent=4)


def main():
    (training_args, test_args, args), config = parse_args(
        (TrainingArguments, TestArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)),
        no_dict=(TrainingArguments,),
        return_config=True
    )
    prediction(training_args, args, config=config, **test_args)
    
if __name__ == "__main__":
    main()

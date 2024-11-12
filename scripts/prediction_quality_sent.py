#!/usr/bin/env python
# coding=utf-8

import json
import tqdm
import torch
from transformers import TrainingArguments, PreTrainedTokenizer, AutoTokenizer
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


PROMPT_FORMAT = "Given a long text, and {num_events} events which take place in the long text, each indicated by number identifier [] that represents the shuffled order, (e.g. [0], [2], etc.). Reorder the events according to the original order of the events in the long text. The events should be listed in descending order using identifiers, and the first event in original order should be list first, and the output format should be [] > [], e.g., [0] > [2]. Only response the reorder results, do not say any word or explain.\n\nLong text:\n{content}\nEvents: {events}\n\nGive the reorder results, only give the ordered identifiers of the {num_events} events {answer_format}: "


@dataclass
class TestArguments(GlobalTestArguments):
    output_file: Optional[str] = field(default=None)


def trainQuALITY(context: str, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, question: Optional[str]=None, title: Optional[str]=None, **args):
    context_dataset = ContextDataset(context, tokenizer, title, question=question, **args)
    model = train(context_dataset, tokenizer, training_args, **args)[0]
    model.eval()
    model = torch.compile(model)
    return model


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
    
    tokenizer = AutoTokenizer.from_pretrained(args['tokenizer_name_or_path'])
    tokenizer.pad_token = tokenizer.eos_token
    for sample_id, sample in enumerate(tqdm.tqdm(samples, desc="Prediction")):
        if sample_id < len(results) - 1:
            continue
        printGPU(f"Before training")
        context = sample['content']
        summaries = sample['summaries']
        #answers = sample['answers']
        prompt = PROMPT_FORMAT.format_map({
            'num_events': len(summaries),
            'events': '\n'.join(f"[{i + 1}]: {summaries[i]}" for i in range(len(summaries))),
            'content': context,
            'answer_format': ' < '.join(['[]'] * len(summaries))
        })
        if args['append_question']:
            model = trainQuALITY(context, tokenizer, training_args, events=summaries, **args)
        else:
            model = trainQuALITY(context, tokenizer, training_args, **args)
        model.eval()
        torch.cuda.empty_cache()
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user', 'content': prompt},
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
        if input_ids.shape[-1] > model_max_length:
            input_ids = torch.concat((input_ids[:, :model_max_length//2], input_ids[:, -model_max_length//2:]), dim=-1)
        attention_mask = torch.ones_like(input_ids)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
                max_new_tokens=32,
                temperature=0.7,
                use_cache=False
            )[0]
        sample['pred'] = tokenizer.decode(output_ids[input_ids.shape[-1]:], skip_special_tokens=True)
        del model
        torch.cuda.empty_cache()
        results.append(sample)
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

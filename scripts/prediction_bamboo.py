#!/usr/bin/env python
# coding=utf-8
import logging
import tqdm
import json
import torch
import os
from transformers import (
    TrainingArguments,
    PreTrainedTokenizer
)
from dataclasses import dataclass, field
from typing import Optional
from long_ttt.ttt_args import ModelArguments, CustomTrainingArguments, DataTrainingArguments, GlobalTestArguments, parse_args
from long_ttt.train import train
from long_ttt.context_dataset import ContextDataset
from long_ttt.model import load_tokenizer


PROMPT_FORMAT = "Given a long text, and {num_events} events which take place in the long text, each indicated by number identifier [] that represents the shuffled order, (e.g. [0], [2], etc.). Reorder the events according to the original order of the events in the long text. The events should be listed in descending order using identifiers, and the first event in original order should be list first, and the output format should be [] > [], e.g., [0] > [2]. Only response the reorder results, do not say any word or explain.\n\nLong text:\n{content}\nEvents: {events}\n\nGive the reorder results, only give the ordered identifiers of the {num_events} events {answer_format}: "


@dataclass
class TestArguments(GlobalTestArguments):
    output_file: Optional[str] = field(default=None)


def Bamboo_train(full_text: str, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, **kwargs):
    dataset = ContextDataset(full_text, tokenizer, **kwargs)
    return train(dataset, tokenizer, training_args, **kwargs)


def generate(model, tokenizer, input_data, model_max_length):
    messages = [
        {'role': 'user', 'content': input_data},
    ]
    input_ids = torch.LongTensor(tokenizer.apply_chat_template(messages, add_generation_prompt=True))[None, :].to(model.device)
    if len(input_ids[0]) > model_max_length:
        input_ids = torch.cat((input_ids[:, :model_max_length//2], input_ids[:, -model_max_length//2:]), dim=1)
    mask_attention = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    output = model.generate(
        input_ids=input_ids,
        attention_mask=mask_attention,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=terminators,
        max_new_tokens=32,
        temperature=0.7,
        use_cache=False,
    )
    response = tokenizer.decode(
        output[0][len(input_ids[0]) :], skip_special_tokens=True
    )
    return response


def prediction(dataset: list[dict], training_args: TrainingArguments, args: dict, output_file: str, num_resumed: int=0, **kwargs):
    logging.warning(f"Unused arguments: {list(kwargs.keys())}")
    tokenizer = load_tokenizer(args['model_name_or_path'])
    model_max_length = args['model_max_length']
    for index, sample in enumerate(tqdm.tqdm(dataset, total=len(dataset), desc="Predicting")):
        if index < num_resumed:
            continue
        summaries = sample["summaries"]
        prompt = PROMPT_FORMAT.format_map({
            'num_events': len(summaries),
            'events': '\n'.join(f"[{i + 1}]: {summaries[i]}" for i in range(len(summaries))),
            'content': sample['content'],
            'answer_format': ' < '.join(['[]'] * len(summaries))
        })
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user', 'content': prompt},
        ]
        if args['append_question']:
            model = Bamboo_train(sample['content'], tokenizer, training_args, events=summaries, **args)[0]
        else:
            model = Bamboo_train(sample['content'], tokenizer, training_args, **args)[0]
        torch.cuda.empty_cache()

        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if len(input_ids) > model_max_length:
            input_ids = input_ids[:model_max_length//2] + input_ids[-model_max_length//2:]
        input_ids = torch.LongTensor(input_ids)[None, :]
        mask_attention = torch.ones_like(input_ids)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        output = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=mask_attention.to(model.device),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=terminators,
            max_new_tokens=32,
            temperature=0.7,
            use_cache=False,
        )
        response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        sample['pred'] = response
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")

def main():
    training_args, test_args, args = parse_args(
        (TrainingArguments, TestArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)),
        no_dict=(TrainingArguments,)
    )
    input_file = test_args.pop('input_file')
    overwrite = test_args.pop('overwrite')
    if not overwrite and os.path.exists(test_args['output_file']):
        with open(test_args['output_file'], 'r') as f:
            num_resumed = len(f.readlines())
    else:
        if os.path.exists(test_args['output_file']):
            os.remove(test_args['output_file'])
        num_resumed = 0
    debug_size = args.pop('debug_size')
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f.readlines()]
    if debug_size is not None:
        dataset = dataset[:debug_size]
    prediction(dataset, training_args, args, num_resumed=num_resumed, **test_args)

if __name__ == "__main__":
    main()

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
from long_ttt.context_dataset import ContextDataset
from long_ttt.utils import printGPU
from typing import Optional
import os


@dataclass
class TestArguments(GlobalTestArguments):
    output_file: Optional[str] = field(default=None)


PROMPT_FORMAT = "Given a long text, and {num_events} events which take place in the long text. \n\nLong text:\n{content}\nThe output format should be [] > [], e.g., [0] > [2]. Only response the reorder results, do not say any word or explain. Question: {question}\n\nGive the reorder results, only give the ordered identifiers of the {num_events} events {answer_format}: "


def LooGLEtrain(full_text: str, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, **kwargs):
    dataset = ContextDataset(full_text, tokenizer, **kwargs)
    print('!' * 100, len(dataset))
    print(dataset[0]['input_ids'].shape)
    return train(dataset, tokenizer, training_args, **kwargs)


def prediction(training_args: TrainingArguments, args: dict, output_file: str, input_file: str="", overwrite: bool=True, config: Optional[dict]=None, **kwargs):
    # resume from checkpoint
    results = [] if config is None else [config]
    if not overwrite and os.path.exists(output_file):
        logging.info(f"Detect existing output file {output_file}. Resume from the checkpoint.")
        with open(output_file, 'r') as f:
            results = json.load(f)
        logging.info(f"Load the results of {len(results)} samples.")
    num_resumed = len(results)
    # load the data
    with open(input_file, 'r') as f:
        data = json.load(f)
    model_max_length = args['model_max_length']
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args['model_name_or_path'])
    tokenizer.pad_token = tokenizer.eos_token
    # eval loop
    for index, sample in enumerate(tqdm.tqdm(data, total=len(data), desc="Predicting")):
        if index < num_resumed:
            continue
        question = sample["question"]
        prompt = PROMPT_FORMAT.format_map({
            'num_events': len(sample['answers']),
            'question': question,
            'content': sample['content'],
            'answer_format': ' < '.join(['[]'] * len(sample['answers']))
        })
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user', 'content': prompt},
        ]
        if args['append_question']:
            model = LooGLEtrain(sample['content'], tokenizer, training_args, question=question, **args)[0]
        else:
            model = LooGLEtrain(sample['content'], tokenizer, training_args, **args)[0]
        model.eval()
        torch.cuda.empty_cache()

        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if len(input_ids) > model_max_length:
            input_ids = input_ids[:model_max_length//2] + input_ids[-model_max_length//2:]
        input_ids = torch.LongTensor(input_ids)[None, :]
        mask_attention = torch.ones_like(input_ids)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids.to(model.device),
                attention_mask=mask_attention.to(model.device),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
                max_new_tokens=128,
                temperature=0.7,
                use_cache=False,
            )
        response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        sample['pred'] = response
        results.append(sample)
        del model, output, input_ids, mask_attention
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)


def main():
    (training_args, test_args, args), config = parse_args(
        (TrainingArguments, TestArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)),
        no_dict=(TrainingArguments,),
        return_config=True
    )
    prediction(training_args, args, config=config, **test_args)
    
if __name__ == "__main__":
    main()

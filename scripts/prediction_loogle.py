#!/usr/bin/env python
# coding=utf-8

import json
import tqdm
import torch
from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass, field
from long_ttt.train import LooGLEtrain
from long_ttt.ttt_args import ModelArguments, CustomTrainingArguments, DataTrainingArguments, GlobalTestArguments
from long_ttt.utils import get_average_attention
from typing import Optional
import os


@dataclass
class TestArguments(GlobalTestArguments):
    output_file: Optional[str] = field(default=None)
    dataset_name: Optional[str] = field(default=None)
    prepend_input: bool = field(default=True)
    recite_first: bool = field(default=False)


def parse_args():
    parser = HfArgumentParser([TrainingArguments, TestArguments, ModelArguments, CustomTrainingArguments, DataTrainingArguments])
    args = {}
    training_args, test_args, *other_args = parser.parse_args_into_dataclasses()
    for class_args in other_args:
        args.update(vars(class_args))
    return training_args, dict(vars(test_args)), args


def prediction_long(training_args: TrainingArguments, args: dict, output_file: str, prepend_input: bool=True, recite_first: bool=False, compute_attention: bool=False, attention_output_dir: Optional[str]=None, input_file: str=""):
    model_max_length = args['model_max_length']
    # load dataset
    debug_size = args.pop('debug_size')
    with open(input_file, "r") as f:
        samples = [json.loads(line) for line in f]
        if debug_size is not None:
            samples = samples[:debug_size]
    # prediction
    results = []
    for sample in samples:
        sample["qa_pairs"] = eval(sample["qa_pairs"])
        model, tokenizer = LooGLEtrain(sample, training_args, **args)
        for param in model.parameters():
            param.grad = None
        from long_ttt.utils import printGPU
        printGPU("Eval")
        for i, qa_pair in enumerate(tqdm.tqdm(sample["qa_pairs"])):
            torch.cuda.empty_cache()
            prompts = []
            prompts += [
                f"Please answer the following question only based on \"{sample['title']}\"."
            ]
            if prepend_input:
                prompts += [
                    f"This is part of the texts from \"{sample['title']}\": \"{sample['input']}\""
                ]
            if recite_first:
                prompts += [
                    f"Please recite the facts from \"{sample['title']}\" that support your answer before answering the question according to the facts.",
                    f"Question: {qa_pair['Q']}",
                    f"Please answer in the following format: \"Evidence: <facts>. Answer: <answer>\". Do not output anything else.",
                ]
            else:
                prompts += [
                    f"Question: {qa_pair['Q']}",
                ]
                
            messages = [
                {'role': 'system', 'content': "You are a helpful assistant. "},
                {'role': 'user', 'content': '\n'.join(prompts)}
            ]
            with torch.no_grad():
                input_ids = torch.LongTensor(tokenizer.apply_chat_template(messages, add_generation_prompt=True))[None, :]
                if prepend_input and len(input_ids[0]) > model_max_length:
                    input_ids = torch.cat((input_ids[:, :model_max_length//2], input_ids[:, -model_max_length//2:]), dim=1)
                mask_attention = torch.ones(input_ids.shape, dtype=torch.long)
                terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    attention_mask=mask_attention,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=terminators,
                    do_sample=False,
                    temperature=1.0,
                    num_beams=1,
                    repetition_penalty=1.1,
                    output_attentions=compute_attention,
                    return_dict_in_generate=True,
                )
                output = outputs.sequences
                attentions = outputs.attentions[-1]
                if compute_attention:
                    get_average_attention(tokenizer, attentions, input_ids, qa_pair['S'], os.path.join(attention_output_dir, f"attn_{sample['title']}_{i}.png"))
                output_model = output[0][input_ids.shape[-1]:]
                pred = tokenizer.decode(output_model, skip_special_tokens=True)
                qa_pair['pred'] = pred
            del output, attentions, input_ids, output_model
        results.append(sample)
        del model, tokenizer
        torch.cuda.empty_cache()
        printGPU("End of task")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)


def prediction_short(training_args: TrainingArguments, args: dict, output_file: str, prepend_input: bool=True, recite_first: bool=False, compute_attention: bool=False, input_file: str=""):
    model_max_length = args['model_max_length']
    with open(input_file, "r") as f:
        samples = [json.loads(line) for line in f]
    
    results = []
    for sample in samples:
        sample["qa_pairs"] = eval(sample["qa_pairs"])
        model, tokenizer = LooGLEtrain(datapoint=sample, training_args=training_args, **args)
        for qa_pair in tqdm.tqdm(sample["qa_pairs"]):
            prompts = []
            prompts += [
                f"Please answer the following question only based on \"{sample['title']}\"."
            ]
            if prepend_input:
                prompts += [
                    f"This is part of the texts from \"{sample['title']}\": \"{sample['input']}\""
                ]
            if recite_first:
                prompts += [
                    f"Please recite the facts from \"{sample['title']}\" that support your answer before answering the question according to the facts.",
                    f"Question: {qa_pair['Q']}",
                    f"Please answer in the following format: \"Evidence: <facts>. Answer: <answer>\". Do not output anything else.",
                ]
            else:
                prompts += [
                    f"Question: {qa_pair['Q']}",
                ]
                
            messages = [
                {'role': 'system', 'content': "You are a helpful assistant. "},
                {'role': 'user', 'content': '\n'.join(prompts)}
            ]
            with torch.no_grad():
                input_ids = torch.LongTensor(tokenizer.apply_chat_template(messages, add_generation_prompt=True))[None, :].to(model.device)
                if prepend_input and len(input_ids[0]) > model_max_length:
                    input_ids = torch.cat((input_ids[:, :model_max_length//2], input_ids[:, -model_max_length//2:]), dim=1)
                mask_attention = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
                terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    attention_mask=mask_attention,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=terminators,
                    do_sample=False,
                    temperature=1.0,
                    num_beams=1,
                    repetition_penalty=1.1,
                    output_hidden_states=compute_attention,
                    return_dict_in_generate=True,
                )
                output = outputs.sequences
                if compute_attention:
                    qa_pair['S_attn'] = get_average_attention(tokenizer, outputs.attentions[-1], input_ids, qa_pair['S'])
                output_model = output[0][input_ids.shape[-1]:]
                pred = tokenizer.decode(output_model, skip_special_tokens=True)
                qa_pair['pred'] = pred
        results.append(sample)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)


def main():
    training_args, test_args, args = parse_args()
    dataset_name = test_args.pop('dataset_name')
    if test_args['attention_output_dir'] is not None:
        os.makedirs(test_args['attention_output_dir'], exist_ok=True)
    if dataset_name == "long_qa":
        prediction_long(training_args, args, **test_args)
    elif dataset_name == "short_qa":
        prediction_short(training_args, args, **test_args)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

if __name__ == "__main__":
    main()
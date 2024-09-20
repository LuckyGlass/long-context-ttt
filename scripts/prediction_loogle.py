#!/usr/bin/env python
# coding=utf-8

import json
import tqdm
import torch
from transformers import TrainingArguments, HfArgumentParser, AutoTokenizer
from dataclasses import dataclass, field
from long_ttt.train import train
from long_ttt.ttt_args import (
    ModelArguments,
    CustomTrainingArguments,
    DataTrainingArguments,
    GlobalTestArguments,
    parse_args
)
from long_ttt.context_dataset import ContextDataset
from long_ttt.utils import get_average_attention
from typing import Optional
import os


@dataclass
class TestArguments(GlobalTestArguments):
    output_file: Optional[str] = field(default=None)
    dataset_name: Optional[str] = field(default=None)
    prepend_input: bool = field(default=True)
    recite_first: bool = field(default=False)


def LooGLEtrain(datapoint: dict, training_args: TrainingArguments, **kwargs):
    """Fine-tune the model and the corresponding tokenizer on a LooGLE task.
    Args:
        datapoint (dict): a LooGLE-style datapoint, containing `input`, `title`, `qa_pairs`.
        tokenizer (PreTrainedTokenizer): a Llama tokenizer (or other tokenizers with chat template).
        training_args (TrainingArguments): transformers-style training arguments, used for the trainer.
        gather_batches (bool): OPTIONAL, default to `False`; if `gather_batches=True`, it will force the trainer to update the model only once every epoch; it may lead to more stable gradients.
        use_lora (bool): OPTIONAL, default to `False`; whether to use LoRA.
        lora_rank (int): OPTIONAL, default to `None`; assign it when `use_lora=True`.
        full_ft (bool): OPTIONAL, default to `False`; whether to full-fine-tune the model.
        load_in_4bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        load_in_8bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        cache_dir (str): OPTIONAL, default to `None`.
        model_revision (str): OPTIONAL, default to `"main"`.
        use_auth_token (bool): OPTIONAL, default to `False`.
        model_max_length (int): OPTIONAL; the texts will be clipped or padded to model_max_length tokens.
        block_size (int): OPTIONAL; the number of tokens in a block; a block is the unit of segments and offsets.
        len_segment (int): OPTIONAL; the number of units in a segment; the article is divided into segments.
        len_offset (int): OPTIONAL; the number of units per offset; it determines the offset from one segment to the next one.
        prepend_title (bool): OPTIONAL; whether to prompt the model with the title.
        sent_token (bool): OPTIONAL; whether to insert a `<|reserved_special_token_249|>` between each two sentences; if enabled, the model must be trained to recognize this token.
    Returns:
        model_tokenizer_pair (tuple[PreTrainedModel, PreTrainedTokenizer]): the fine-tuned model and the corresponding tokenizer.
    """
    tokenizer_kwargs = {
        "cache_dir": kwargs.get("cache_dir", None),
        "use_auth_token": kwargs.get("use_auth_token", False),
        "revision": kwargs.get("model_revision", "main"),
        "use_fast": True, 
    }
    tokenizer = AutoTokenizer.from_pretrained(kwargs['model_name_or_path'], **tokenizer_kwargs)
    dataset = ContextDataset(datapoint['input'], tokenizer, title=datapoint['title'], **kwargs)
    return train(dataset, tokenizer, training_args, **kwargs)


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
                attentions = None if outputs.attentions is None else outputs.attentions[-1]
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


def prediction_short(training_args: TrainingArguments, args: dict, output_file: str, prepend_input: bool=True, recite_first: bool=False, compute_attention: bool=False, attention_output_dir: Optional[str]=None, input_file: str=""):
    model_max_length = args['model_max_length']
    with open(input_file, "r") as f:
        samples = [json.loads(line) for line in f]
    debug_size = args.pop('debug_size')
    with open(input_file, "r") as f:
        samples = [json.loads(line) for line in f]
        if debug_size is not None:
            samples = samples[:debug_size]
    
    results = []
    for sample in samples:
        sample["qa_pairs"] = eval(sample["qa_pairs"])
        model, tokenizer = LooGLEtrain(datapoint=sample, training_args=training_args, **args)
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
                    output_attentions=compute_attention,
                    return_dict_in_generate=True,
                )
                output = outputs.sequences
                attentions = None if outputs.attentions is None else outputs.attentions[-1]
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


def main():
    training_args, test_args, args = parse_args(
        (TrainingArguments, TestArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)),
        no_dict=(TrainingArguments,)
    )
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

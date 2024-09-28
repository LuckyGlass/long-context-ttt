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
from long_ttt.context_dataset import ContextDataset, apply_qa_template
from long_ttt.utils import get_average_attention, printGPU
from typing import Optional
import os


@dataclass
class TestArguments(GlobalTestArguments):
    output_file: Optional[str] = field(default=None)


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
    tokenizer.pad_token = tokenizer.eos_token
    dataset = ContextDataset(datapoint['input'], tokenizer, title=datapoint['title'], **kwargs)
    return train(dataset, tokenizer, training_args, **kwargs)


@torch.no_grad()
def pred_batch(model, tokenizer, index: int, qa_pairs: list[dict], title: str, context: str, model_max_length: int=8000, prepend_input: bool=False, recite_first: bool=False, compute_attention: bool=False, attention_output_dir: Optional[str]=None):
    # Batch
    list_input_ids = []
    for qa_pair in qa_pairs:
        prompt = apply_qa_template(
            question=qa_pair['Q'],
            title=title,
            context=context,
            prepend_title=True,
            prepend_input=prepend_input,
            recite_first=recite_first,
            return_answer=False
        )
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant. "},
            {'role': 'user', 'content': '\n'.join(prompt)}
        ]
        input_ids = torch.LongTensor(tokenizer.apply_chat_template(messages, add_generation_prompt=True)).flatten()
        if prepend_input and input_ids.shape[0] > model_max_length:
            input_ids = torch.cat((input_ids[:model_max_length//2], input_ids[-model_max_length//2:]))
        if input_ids.shape[0] < model_max_length:
            input_ids = torch.cat((input_ids, torch.LongTensor([tokenizer.pad_token_id] * (model_max_length - input_ids.shape[0]))))
        list_input_ids.append(input_ids)
    input_ids = torch.stack(list_input_ids)
    # Forward
    mask_attention = torch.ne(input_ids, tokenizer.pad_token_id)
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
    output_ids = outputs.sequences
    attentions = None if outputs.attentions is None else outputs.attentions[-1]
    if compute_attention:
        get_average_attention(tokenizer, attentions, input_ids, qa_pair['S'], os.path.join(attention_output_dir, f"attn_{title}_{index}.png"))
    output_ids = output_ids[:, input_ids.shape[-1]:]
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    for qa_pair, pred in zip(qa_pairs, preds):
        qa_pair['pred'] = pred


def prediction(training_args: TrainingArguments, args: dict, output_file: str, compute_attention: bool=False, eval_batch_size: int=1, input_file: str="", **kwargs):
    model_max_length = args['model_max_length']
    with open(input_file, "r") as f:
        samples = [json.loads(line) for line in f]
    debug_size = args.pop('debug_size')
    with open(input_file, "r") as f:
        samples = [json.loads(line) for line in f]
        if debug_size is not None:
            samples = samples[:debug_size]
    
    results = []
    for sample in tqdm.tqdm(samples, desc="Prediction"):
        torch.cuda.empty_cache()
        printGPU(f"Before training")
        sample["qa_pairs"] = eval(sample["qa_pairs"])
        model, tokenizer = LooGLEtrain(datapoint=sample, training_args=training_args, **args)
        if not compute_attention:
            model.eval()  # Setting the model to eval mode to speed up inference. However, on eval mode the model can't output attentions.
        for param in model.parameters():
            param.grad = None
        printGPU(f"Eval with {len(sample['qa_pairs'])} samples")
        for i, st_pos in enumerate(tqdm.tqdm(range(0, len(sample['qa_pairs']), eval_batch_size), desc="Sample")):
            torch.cuda.empty_cache()
            qa_pairs = sample['qa_pairs'][st_pos:st_pos+eval_batch_size]
            pred_batch(model, tokenizer, i, qa_pairs, sample['title'], sample['input'], model_max_length=model_max_length, compute_attention=compute_attention, **kwargs)
        results.append(sample)
        del model, tokenizer
        with open(output_file, "w+") as f:
            json.dump(results, f, indent=4)


def main():
    training_args, test_args, args = parse_args(
        (TrainingArguments, TestArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)),
        no_dict=(TrainingArguments,)
    )
    test_args['recite_first'] = args['recite_first']
    test_args['prepend_input'] = args['prepend_input']
    if test_args['attention_output_dir'] is not None:
        os.makedirs(test_args['attention_output_dir'], exist_ok=True)
    prediction(training_args, args, **test_args)
    
if __name__ == "__main__":
    main()

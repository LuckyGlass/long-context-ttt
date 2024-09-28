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
from typing import Optional
import os
from long_ttt.utils import printGPU
from long_ttt.model import load_tokenizer

@dataclass
class TestArguments(GlobalTestArguments):
    output_file: Optional[str] = field(default=None)


def LongbenchTrain(datapoint: dict, training_args: TrainingArguments, **kwargs):
    tokenizer = load_tokenizer(kwargs['model_name_or_path'])
    dataset = ContextDataset(datapoint['context'], tokenizer, **kwargs)
    return train(dataset, tokenizer, training_args, **kwargs)


def get_prediction(training_args: TrainingArguments, args: dict, output_file: str, prepend_input: bool=True, recite_first: bool=False, compute_attention: bool=False, attention_output_dir: Optional[str]=None, input_file: str=""):
    model_max_length = args['model_max_length']
    dataset2prompt = json.load(open("/scratch/nlp/lijiaqi/long-context-ttt/scripts/longbench/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("/scratch/nlp/lijiaqi/long-context-ttt/scripts/longbench/dataset2maxlen.json", "r"))

    # load dataset
    debug_size = args.pop('debug_size')
    for root, dirs, files in os.walk(input_file):
        for dataset in files:
            print('---------------------------------------------------------------------------'+input_file+dataset)
            with open(input_file+dataset, 'r') as f:
                samples = [json.loads(line) for line in f]
                if debug_size is not None:
                    samples = samples[:debug_size]
            prompt_ins = dataset2prompt[dataset.replace('.jsonl','')]
            max_gen = dataset2maxlen[dataset.replace('.jsonl','')]

            #prediction
            results = []
            c = 0
            for sample in samples:
                model, tokenizer = LongbenchTrain(sample, training_args, **args)
                for param in model.parameters():
                    param.grad = None
                torch.cuda.empty_cache()
                printGPU("Eval")

                torch.cuda.empty_cache()
                prompts = []
                if prepend_input:
                    prompts += [f"Here is the [context] for the task: {sample['context']}" ]
                if recite_first:
                    prompts += [f"Please recite the facts from the context that support your answer before answering the question according to the facts."]

                prompts += [f"Question: {sample['input']}" ]
                prompts += [prompt_ins]
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
                        max_new_tokens= max_gen,
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
                    output_model = output[0][input_ids.shape[-1]:]
                    pred = tokenizer.decode(output_model, skip_special_tokens=True)
                    sample['pred'] = pred
                del output, attentions, input_ids, output_model
                results.append(sample)
                del model, tokenizer
                torch.cuda.empty_cache()
                printGPU("End of task")
                if not os.path.exists(output_file):
                    os.makedirs(output_file)
                with open(output_file+dataset, "a+", encoding="utf-8") as f:
                    json.dump({"pred": pred, "answers": sample["answers"], "all_classes": sample["all_classes"], "length": sample["length"], "_id": sample["_id"]},f, ensure_ascii=False)
                    f.write('\n')
                c += 1
                print('====================Finish', c)

def main():
    training_args, test_args, args = parse_args(
        (TrainingArguments, TestArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)),
        no_dict=(TrainingArguments,)
    )
    test_args['recite_first'] = args['recite_first']
    test_args['prepend_input'] = args['prepend_input']
    if test_args['attention_output_dir'] is not None:
        os.makedirs(test_args['attention_output_dir'], exist_ok=True)
    
    get_prediction(training_args, args, **test_args)


if __name__ == "__main__":
    main()

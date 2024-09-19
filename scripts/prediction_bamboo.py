#!/usr/bin/env python
# coding=utf-8
import tqdm
import json
import torch
import re
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    PreTrainedTokenizer,
    AutoTokenizer
)
from dataclasses import dataclass, field
from long_ttt.ttt_args import ModelArguments, CustomTrainingArguments, DataTrainingArguments
from long_ttt.train import train
from long_ttt.context_dataset import ContextDataset


@dataclass
class TestArguments:
    input_file: str
    output_file: str
    prompt_name: str
    prompt_path: str


def Bamboo_train(full_text: str, training_args, **kwargs):
    tokenizer_kwargs = {
        "cache_dir": kwargs.get("cache_dir", None),
        "use_auth_token": kwargs.get("use_auth_token", False),
        "revision": kwargs.get("model_revision", "main"),
        "use_fast": True, 
    }
    tokenizer = AutoTokenizer.from_pretrained(kwargs['model_name_or_path'], **tokenizer_kwargs)
    dataset = ContextDataset(full_text, tokenizer, **kwargs)
    return train(dataset, tokenizer, training_args, **kwargs)


def parse_args():
    parser = HfArgumentParser([TrainingArguments, TestArguments, ModelArguments, CustomTrainingArguments, DataTrainingArguments])
    args = {}
    training_args, test_args, *other_args = parser.parse_args_into_dataclasses()
    for class_args in other_args:
        args.update(vars(class_args))
    return training_args, dict(vars(test_args)), args


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


def prediction(dataset: list[dict], training_args: TrainingArguments, args: dict, prompt: dict, prompt_name: str, output_file: str):
    for index, sample in enumerate(tqdm.tqdm(dataset, total=len(dataset), desc="Predicting")):
        if prompt_name == "reportsumsort":
            full_text = sample["content"]
            summaries = sample["summaries"]
            input_text = []
            for i in range(len(summaries)):
                input_text.append("[{}] ```{}```".format(i, summaries[i]))
            input_data = prompt["final_answer"].format(
                content=full_text, events="\n".join(input_text)
            )
            input_text = input_data
            answer = sample["answer"]
            
            # prediction
            model, tokenizer = Bamboo_train(full_text, training_args, **args)
            pred = generate(model, tokenizer, input_text, args['model_max_length'])
            numbers = re.findall(r"\d+", pred)
            numbers = [int(y) for y in numbers]
            processed_pred = numbers
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"pred": processed_pred, "answer": answer, "output": pred})
                    + "\n"
                )

def main():
    training_args, test_args, args = parse_args()
    input_file = test_args.pop('input_file')
    prompt_path = test_args.pop('prompt_path')
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f.readlines()]
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = json.loads(f.read())[test_args['prompt_name']]
    del test_args['compute_attention'], test_args['attention_output_dir']
    prediction(dataset, training_args, args, prompt, **test_args)

if __name__ == "__main__":
    main()

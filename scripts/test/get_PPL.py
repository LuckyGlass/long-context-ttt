"""
Compute PPL for LLMs with full context window (no window shifting nor truncation).
Following https://huggingface.co/docs/transformers/main/en/perplexity#calculating-ppl-with-fixed-length-models
In the default mode, the dataset should contain datapoints in the format
```
{
    'input': ...,
    'output': ...  // OPTIONAL if require_generation=True
}
```
In the chat template mode, the dataset should contain datapoints in the format
```
[
    {'role': "system", 'content': ...},  // OPTIONAL
    {'role': "user", 'content': ...},
    {'role': "assistant", 'content': ...}  // OPTIONAL if require_generation=True
]
```
"""
from dataclasses import dataclass, field
import json
import numpy as np
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    LlamaForCausalLM
)
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from typing import Optional


class MyDataset(Dataset):
    def __init__(self, tokenizer, args):
        # Load the dataset
        with open(args.dataset_path, 'r') as f:
            dataset_path_extention = os.path.splitext(args.dataset_path)[-1]
            if dataset_path_extention == '.json':
                data = json.load(f)
            elif dataset_path_extention == '.jsonl':
                data = list(map(json.loads, f.readlines()))
            else:
                raise ValueError(f"Unrecognized extension of dataset path, \'{dataset_path_extention}\'.")
        # Process the dataset
        self.datapoint_list = []
        for datapoint in data:
            if args.apply_chat_template:
                if args.require_generation:
                    input_ids = tokenizer.apply_chat_template(
                        datapoint,
                        add_generation_prompt=True,
                        return_tensors='pt',
                        padding='max_length',
                        truncation=True,
                        max_length=args.input_max_length,
                    ).flatten()
                    attention_mask = input_ids.ne(tokenizer.pad_token_id)
                    self.datapoint_list.append({
                        'input_ids': input_ids,
                        'attention_mask': attention_mask
                    })
                else:
                    input_ids = tokenizer.apply_chat_template(
                        datapoint,
                        add_generation_prompt=True,
                        return_tensors='pt',
                        padding='max_length',
                        truncation=True,
                        max_length=args.model_max_length,
                    ).flatten()
                    input_length = tokenizer.apply_chat_template(
                        datapoint,
                        add_generation_prompt=True,
                        return_tensors='pt'
                    ).shape[-1]
                    attention_mask = input_ids.ne(tokenizer.pad_token_id)
                    target_ids = input_ids.clone()
                    target_ids[:input_length] = -100
                    target_ids[~attention_mask] = -100
                    self.datapoint_list.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'target_ids': target_ids})
            else:
                if args.require_generation:
                    input_ids = tokenizer(
                        datapoint['input'],
                        add_special_tokens=args.add_special_tokens,
                        padding='max_length',
                        truncation=True,
                        max_length=args.input_max_length,
                        return_tensors='pt',
                    ).input_ids.flatten()
                    attention_mask = input_ids.ne(tokenizer.pad_token_id)
                    self.datapoint_list.append({
                        'input_ids': input_ids,
                        'attention_mask': attention_mask
                    })
                else:
                    input_ids = tokenizer(
                        datapoint['input'] + datapoint['output'],
                        add_special_tokens=args.add_special_tokens,
                        padding='max_length',
                        truncation=True,
                        max_length=args.model_max_length,
                        return_tensors='pt',
                    ).input_ids.flatten()
                    input_length = tokenizer(
                        datapoint['input'],
                        add_special_tokens=args.add_special_tokens,
                        return_tensors='pt',
                    ).input_ids.shape[-1]
                    attention_mask = input_ids.ne(tokenizer.pad_token_id)
                    target_ids = input_ids.clone()
                    target_ids[:input_length] = -100
                    target_ids[~attention_mask] = -100
                    self.datapoint_list.append({
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'target_ids': target_ids
                    })
    
    def __len__(self):
        return len(self.datapoint_list)
    
    def __getitem__(self, index):
        return self.datapoint_list[index]


@dataclass
class GenerationArguments:
    model_name_or_path: Optional[str] = field(default=None)
    batch_size: int = field(default=1)
    model_max_length: int = field(default=2048)
    input_max_length: Optional[int] = field(default=None)
    dataset_path: Optional[str] = field(default=None)
    output_path: Optional[str] = field(default=None)
    require_generation: bool = field(default=False)
    apply_chat_template: bool = field(default=False)
    add_special_tokens: bool = field(default=False)
    overwrite: bool = field(default=True)
    
    def __post_init__(self):
        if self.input_max_length is None and self.require_generation:
            raise ValueError('Please assign a value for input_max_length. The input_ids will be padded to that length.')


@torch.no_grad()
def main():
    parser = HfArgumentParser((GenerationArguments,))
    args: GenerationArguments = parser.parse_args_into_dataclasses()[0]
    # Load the model and the tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='auto'
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    # Load the dataset and the dataloader
    dataset = MyDataset(tokenizer, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    # Clean the output file
    if args.overwrite and args.output_path is not None and os.path.exists(args.output_path):
        os.remove(args.output_path)
    # Forward and Compute PPL
    results = []
    sum_nll = 0
    num_valid_tokens = 0
    for batch_id, batch in enumerate(tqdm.tqdm(dataloader)):
        batch_size = batch['input_ids'].shape[0]
        if args.require_generation:
            outputs = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=args.model_max_length-args.input_max_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
            )
            output_ids = outputs.sequences
            attention_mask = output_ids.ne(tokenizer.pad_token_id)
            target_ids = output_ids.clone()
            target_ids[:, :args.input_max_length] = -100
            target_ids[~attention_mask] = -100
            loss = model(input_ids=output_ids, attention_mask=attention_mask, labels=target_ids).loss.cpu().item()
            pred_ids = output_ids[:, args.input_max_length:]
            pred_strs = tokenizer.batch_decode(pred_ids)
            input_strs = tokenizer.batch_decode(batch['input_ids'])
            results.append({
                'batch_id': batch_id,
                'data': [{'input': input_str, 'output': pred_str} for input_str, pred_str in zip(input_strs, pred_strs)],
                'batch_nll': loss
            })
            num_valid_tokens_batch = torch.sum(target_ids.ne(-100)).cpu().item()
            sum_nll += loss * num_valid_tokens_batch
            num_valid_tokens += num_valid_tokens_batch
        else:
            loss = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['target_ids'],
            ).loss.cpu().item()
            input_strs = tokenizer.batch_decode(batch['input_ids'])
            results.append({
                'batch_id': batch_id,
                'data': [{'input': input_str} for input_str in input_strs],
                'batch_nll': loss
            })
            num_valid_tokens_batch = torch.sum(batch['target_ids'].ne(-100)).cpu().item()
            sum_nll += loss * num_valid_tokens_batch
            num_valid_tokens += num_valid_tokens_batch
        if args.output_path is not None:
            output_path_extension = os.path.splitext(args.output_path)[-1]
            if output_path_extension == '.json':
                with open(args.output_path, 'w') as f:
                    json.dump(results, f)
            elif output_path_extension == '.jsonl':
                with open(args.output_path, 'a') as f:
                    f.write(json.dumps(results[-1]) + '\n')
    global_ppl = np.exp(sum_nll / num_valid_tokens)
    results.append({'global_ppl': global_ppl})
    print(f"Global PPL = {global_ppl}")
    if args.output_path is not None:
        output_path_extension = os.path.splitext(args.output_path)[-1]
        if output_path_extension == '.json':
            with open(args.output_path, 'w') as f:
                json.dump(results, f)
        elif output_path_extension == '.jsonl':
            with open(args.output_path, 'a') as f:
                f.write(json.dumps(results[-1]) + '\n')


if __name__ == '__main__':
    main()

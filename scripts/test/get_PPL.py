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
import tqdm
from typing import Optional


LlamaForCausalLM.forward()


@dataclass
class GenerationArguments:
    model_name_or_path: Optional[str] = field(default=None)
    dataset_path: Optional[str] = field(default=None)
    output_path: Optional[str] = field(default=None)
    require_generation: bool = field(default=False)
    apply_chat_template: bool = field(default=False)
    add_special_tokens: bool = field(default=False)
    overwrite: bool = field(default=True)


def main():
    parser = HfArgumentParser((GenerationArguments,))
    args, = parser.parse_args_into_dataclasses()
    # Load the model and the tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # Load the dataset
    with open(args.dataset_path, 'r') as f:
        dataset_path_extention = os.path.split(args.dataset_path)[-1]
        if dataset_path_extention == 'json':
            data = json.load(f)
        elif dataset_path_extention == 'jsonl':
            data = list(map(json.loads, f.readlines()))
        else:
            raise f"Unrecognized extension of dataset path, \'{dataset_path_extention}\'."
    # Clean the output file
    if args.overwrite and args.output_path is not None:
        os.remove(args.output_path)
    # Forward and Compute PPL
    results = []
    nlls = []
    for datapoint in tqdm.tqdm(data):
        if args.require_generation:
            if args.apply_chat_template:
                input_ids = tokenizer.apply_chat_template(datapoint, add_generation_prompt=True, return_tensors='pt')
            else:
                input_ids = tokenizer(datapoint['input'], add_special_tokens=args.add_special_tokens, return_tensors='pt')
            input_ids = input_ids.reshape(1, -1)
            output_ids = model.generate(input_ids)
            target_ids = output_ids.clone()
            target_ids[:, :input_ids.shape[-1]] = -100
            outputs = model(output_ids, labels=target_ids)
            nlls.append(outputs.loss)
        else:
            if args.apply_chat_template:
                input_length = tokenizer.apply_chat_template(datapoint[:-1], add_generation_prompt=True, return_tensors='pt').shape[-1]
                input_ids = tokenizer.apply_chat_template(datapoint, add_generation_prompt=False, return_tensors='pt')
            else:
                input_length = tokenizer(datapoint['input'], add_special_tokens=args.add_special_tokens, return_tensors='pt').input_ids.shape[-1]
                input_ids = tokenizer(datapoint['input'] + datapoint['output'], add_special_tokens=args.add_special_tokens, return_tensors='pt').input_ids
            input_ids = input_ids.reshape(1, -1)
            labels = input_ids.clone()
            labels[:, :input_length] = -100
            outputs = model(input_ids=input_ids, labels=labels)
            nlls.append(outputs.loss)
        results.append({
            'data': datapoint,
            'ppl': np.exp(nlls[-1])
        })
        if args.output_path is not None:
            output_path_extension = os.path.split(args.output_path)[-1]
            if output_path_extension == 'json':
                with open(args.output_path, 'w') as f:
                    json.dump(results, f)
            elif output_path_extension == 'jsonl':
                with open(args.output_path, 'a') as f:
                    f.write(json.dumps(results[-1]) + '\n')
    global_ppl = np.exp(np.mean(nlls))
    results.append({'global_ppl': global_ppl})
    print(f"Global PPL = {global_ppl}")
    if args.output_path is not None:
        output_path_extension = os.path.split(args.output_path)[-1]
        if output_path_extension == 'json':
            with open(args.output_path, 'w') as f:
                json.dump(results, f)
        elif output_path_extension == 'jsonl':
            with open(args.output_path, 'a') as f:
                f.write(json.dumps(results[-1]) + '\n')


if __name__ == '__main__':
    main()

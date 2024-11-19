"""
Used to evaluate running time.
Settings:
1. Baseline: pretrained long-context model (Llama-3.1-8B-Instruct), full ICL.
2. Quantized: pretrained long-context model (Llama-3.1-8B-Instruct), 4-bit, full ICL.
3. TTT: 5-epoch, truncated ICL.
4. SFT+TTT: 5-epoch, truncated ICL, expected to be the same as TTT.
"""
import json
import numpy as np
import time
import torch
import tqdm
from transformers import PreTrainedTokenizer, TrainingArguments
from long_ttt.model import load_tokenizer, load_model
from long_ttt.train import train
from long_ttt.ttt_args import (
    ModelArguments,
    CustomTrainingArguments,
    DataTrainingArguments,
    parse_args
)
from long_ttt.utils import printGPU
from long_ttt.context_dataset import ContextDataset
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestArguments:
    setting: str = field(default='unk')
    output_path: Optional[str] = field(default=None)


def construct_input(tokenizer: PreTrainedTokenizer, length: int) -> torch.Tensor:
    token = tokenizer.convert_tokens_to_ids('Hello')
    prompt = tokenizer("Please generate a very long article.", add_special_tokens=False, return_tensors='pt').flatten()
    padding = torch.LongTensor([token] * (length - prompt.shape[-1]))
    input_ids = torch.concat([padding, prompt])[None, :]
    return input_ids


def single_test(model_name_or_path, tokenizer, length, setting, training_args, **kwargs):
    model_max_length = kwargs['model_max_length']
    model = load_model(model_name_or_path, tokenizer, **kwargs)
    input_ids = construct_input(tokenizer, length).to(model.device)
    gputime_1 = torch.cuda.Event(enable_timing=True)
    gputime_2 = torch.cuda.Event(enable_timing=True)
    if setting in ['Baseline', 'Quantized']:
        # Start the timer
        systime_1 = time.time()
        gputime_1.record()
        # Eval
        model.eval()
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
                max_new_tokens=10,
                use_cache=False,
            )
        # Stop the timer
        systime_2 = time.time()
        gputime_2.record()
        assert output_ids.shape[-1] == input_ids.shape[-1] + 10
    elif setting in ['TTT', 'SFT+TTT']:
        input_str = tokenizer.decode(input_ids.flatten())
        # Start the timer
        systime_1 = time.time()
        gputime_1.record()
        # Eval
        dataset = ContextDataset(input_str, tokenizer, **kwargs)
        model = train(dataset, tokenizer, training_args, **kwargs)
        model.eval()
        if input_ids.shape[-1] > model_max_length:
            input_ids = torch.concat([input_ids[:, :model_max_length//2], input_ids[:, -model_max_length//2:]], dim=-1)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
                max_new_tokens=10,
                use_cache=False,
            )
        # Stop the timer
        systime_2 = time.time()
        gputime_2.record()
        assert output_ids.shape[-1] == input_ids.shape[-1] + 10
        del dataset
    torch.cuda.synchronize()
    systime_per_token = (systime_2 - systime_1) / 10
    gputime_per_token = (gputime_1.elapsed_time(gputime_2) / 1000) / 10
    del model, input_ids, output_ids
    return systime_per_token, gputime_per_token


def test(setting: str, output_path: str, model_name_or_path: str, training_args: TrainingArguments, **kwargs):
    tokenizer = load_tokenizer(model_name_or_path)
    test_lengths = [i * 1000 for i in range(1, 101)]
    for length in tqdm.tqdm(test_lengths, desc='Length'):
        systime = []
        gputime = []
        for _ in range(5):
            torch.cuda.empty_cache()
            printGPU()
            a, b = single_test(model_name_or_path, tokenizer, length, setting, training_args, **kwargs)
            systime.append(a)
            gputime.append(b)
        with open(output_path, 'a') as f:
            json.dump({'length': length, 'systime': np.mean(systime), 'gputime': np.mean(gputime)}, f)
            f.write('\n')


def main():
    args, training_args = parse_args(((TestArguments, ModelArguments, CustomTrainingArguments, DataTrainingArguments), TrainingArguments), no_dict=TrainingArguments)
    test(training_args=training_args, **args)


if __name__ == '__main__':
    main()

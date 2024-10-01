import os
import math
import torch
import json
import datasets
import pickle

import numpy as np

from fastchat.model import get_conversation_template

from glob import glob
from typing import List, Optional
from tqdm import tqdm
from transformers import TrainingArguments
from transformers.utils import logging
from dataclasses import dataclass, field, asdict

from needle_base.utils import DefaultDataCollator, FileLogger, makedirs

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from long_ttt.model import load_tokenizer
from long_ttt.context_dataset import ContextDataset
from long_ttt.train import train
from long_ttt.ttt_args import (
    GlobalTestArguments,
    CustomTrainingArguments,
    DataTrainingArguments,
    ModelArguments,
    parse_args
)
import os

logger = logging.get_logger(__name__)


@dataclass
class TestArgs(GlobalTestArguments):
    haystack_path: str = field(
        default="long-llm:needle/PaulGrahamEssays",
        metadata={'help': 'The context for evaluation.'}
    )
    result_dir: str = field(
        default="",
        metadata={'help': 'The base directory for saving results and logs.'}
    )

    min_length: int = field(
        default=8192,
        metadata={'help': 'Minimum context length in evaluation.'}
    )
    max_length: int = field(
        default=131072,
        metadata={'help': 'Maximum context length in evaluation.'}
    )
    num_length_interval: int = field(
        default=10,
        metadata={'help': 'Number of invervals between min_length and max_length.'}
    )
    test_length: List[int] = field(
        default=None,
        metadata={'help': 'Specified evaluation lengths.'}
    )

    min_depth: float = field(
        default=0,
        metadata={'help': 'Minimum pass key depth in the context.'}
    )
    max_depth: float = field(
        default=100,
        metadata={'help': 'Maximum pass key depth in the context.'}
    )
    num_depth_interval: int = field(
        default=6,
        metadata={'help': 'Number of invervals between min_depth and max_depth.'}
    )
    test_depth: List[int] = field(
        default=None,
        metadata={'help': 'Specified evaluation depths.'}
    )

    needle: str = field(
        default="\n\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n",
        metadata={'help': 'The needle content'}
    )
    prompt: str = field(
        default='\n\nWhat is the best thing to do in San Francisco?\nAnswer:',
        metadata={'help': 'The needle content'}
    )

    chat_template: str = field(
        default="vicuna",
        metadata={'help': 'Instruction template name in fastchat.'}
    )
    gpt_eval: bool = field(
        default=False,
        metadata={'help': 'Use GPT4 to evaluate accuracy.'}
    )
    cpu: bool = field(
        default=False,
        metadata={'help': 'Use cpu?'}
    )


    zh: bool = field(
        default=False,
        metadata={'help': 'Eval Chinese Text.'}
    )
    
    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)


def generate_sample(
    tokenizer, 
    context, 
    context_length, 
    needle_depth, 
    needle="\n\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n", 
    prompt='\n\nWhat is the best thing to do in San Francisco?\nAnswer:',
    chat_template="vicuna",
    zh=False,
):
    """ It's modified for TTT use.
    Args:
        tokenizer:
    Returns:
        - inputs, the full input string (chat template applied).
        - context, the context (the context documents and the needle).
        - needle.
    """
    if zh:
        num_words = len(context)
    else:
        num_words = len(context.split())
    if context_length > num_words:
        context = context * math.ceil(context_length / num_words)

    if zh:
        description = "以下上下文中隐藏着重要信息。找到并记住这些信息。我会问你关于其中重要信息的问题。\n"
    else:
        description = "There is an important infomation hidden in the following context. Find the information and memorize it. I will quiz you about the important information there.\n"

    description_input_ids = tokenizer.encode(description, add_special_tokens=False)
    needle_input_ids = tokenizer.encode(needle, add_special_tokens=False)
    prompt_input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    description_length = len(description_input_ids)
    needle_length = len(needle_input_ids)
    prompt_length = len(prompt_input_ids)

    # must leave room for information and prompt
    minimum_pos = description_length
    maximum_pos = context_length - prompt_length - needle_length - 1
    if minimum_pos > context_length or maximum_pos < 0:
        raise ValueError(f"The length {context_length} is too small. Please increase interval!")

    needle_pos = minimum_pos + round((maximum_pos - minimum_pos) * needle_depth / 100)
    
    context_input_ids = tokenizer.encode(context, max_length=context_length - description_length - needle_length - prompt_length, truncation=True, add_special_tokens=False)

    input_ids = sum([description_input_ids, context_input_ids[:needle_pos], needle_input_ids, context_input_ids[needle_pos:], prompt_input_ids], [])
    context_ids = sum([context_input_ids[:needle_pos], needle_input_ids, context_input_ids[needle_pos:]], [])
    inputs = tokenizer.decode(input_ids)
    context_return = tokenizer.decode(context_ids)

    if chat_template != "none":
        conv = get_conversation_template(chat_template)
        conv.append_message(conv.roles[0], inputs)
        conv.append_message(conv.roles[1], None)
        inputs = conv.get_prompt()

    return inputs, context_return, needle


@torch.no_grad()
def main():
    args, training_args, ttt_args = parse_args((TestArgs, TrainingArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)), no_dict=(TrainingArguments, TestArgs))

    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

    tokenizer = load_tokenizer(ttt_args['model_name_or_path'])
    
    # Needle in a haystack generation configs
    if args.test_length is None:
        test_lengths = np.linspace(args.min_length, args.max_length, args.num_length_interval, endpoint=True).astype(int).tolist()
    else:
        test_lengths = args.test_length
    if args.test_depth is None:
        test_depths = np.linspace(args.min_depth, args.max_depth, args.num_depth_interval, endpoint=True).astype(int).tolist()
    else:
        test_depths = args.test_depth

    # Read context datasets
    if os.path.isfile(args.haystack_path):
        with open(args.haystack_path) as f:
            context = f.read().strip()
    elif os.path.isdir(args.haystack_path):
        context = ""
        num_tokens = 0
        for file in glob(f"{args.haystack_path}/*.txt"):
            with open(file, 'r') as f:
                this_file_context = f.read()
                num_tokens += len(tokenizer.encode(this_file_context, add_special_tokens=False))
                context += this_file_context
                if num_tokens > max(test_lengths):
                    break
    else:
        raise ValueError(f"Cannot find haystack: {args.haystack_path}")
    # Load or create the input cache
    pickle_name = os.path.join(args.haystack_path, f"{args.min_length}-{args.max_length}.pickle")
    print(pickle_name)
    if os.path.exists(pickle_name):
        with open(pickle_name, "rb") as handle:
            all_inputs = pickle.load(handle)
    else:
        all_inputs = []
        for length in tqdm(test_lengths, desc="Constructing Data"):
            for depth in test_depths:
                inputs, context_return, needle = generate_sample(
                    tokenizer=tokenizer, 
                    context=context,
                    context_length=length, 
                    needle_depth=depth,
                    needle=args.needle,
                    prompt=args.prompt,
                    chat_template=args.chat_template,
                    zh=args.zh
                )
                all_inputs.append({'inputs': inputs, 'context_return': context_return, 'needle': needle, 'length': length, 'depth': depth})
        with open(pickle_name, 'wb') as handle:
            pickle.dump(all_inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Create the dataset and the dataloader
    dataset = datasets.Dataset.from_list(all_inputs)
    dataloader = torch.utils.data.DataLoader(
        dataset.remove_columns(['length', 'depth', 'needle']), 
        batch_size=1, 
        collate_fn=DefaultDataCollator(tokenizer),
        pin_memory=not args.cpu,
    )

    # Forward
    all_outputs = []
    for x in tqdm(dataloader, desc="Evaluating"):
        torch.cuda.empty_cache()
        inputs = x.pop("inputs")
        context_return = x.pop('context_return')[0]
        print(context_return)
        print(type(context_return))
        context_dataset = ContextDataset(
            context=context_return,
            tokenizer=tokenizer,
            **ttt_args
        )
        model = train(context_dataset, tokenizer, training_args, **ttt_args)[0]

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=1,
            do_sample=False,
            temperature=1.,
            pad_token_id=tokenizer.pad_token_id,
        )
        outputs = outputs[:, inputs['input_ids'].shape[1]:].contiguous()

        all_outputs.extend(outputs.tolist())

    # Decode and save the outputs and the config
    results = {l: {d: [] for d in test_depths} for l in test_lengths}

    all_outputs = tokenizer.batch_decode(all_outputs, skip_special_tokens=True)
    all_lengths = dataset['length']
    all_depths = dataset['depth']
    all_needles = dataset['needle']

    for l, d, n, o in zip(all_lengths, all_depths, all_needles, all_outputs):
        if args.zh:
            n = n.replace('\\n', '\n')
            o = o.replace('\\n', '\n')
        results[l][d].append({'target': n.replace('\n', ''), 'prediction': o.split('\n')[0].replace('\n', '')})

    with open(makedirs(os.path.join(result_dir, "results.json")), "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
    args.save(os.path.join(result_dir, "config.json"))


if __name__ == "__main__":
    main()

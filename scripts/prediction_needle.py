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
import torch.utils
import torch.utils.data
from tqdm import tqdm
from transformers import TrainingArguments, PreTrainedTokenizer
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
import time

logger = logging.get_logger(__name__)


@dataclass
class TestArgs(GlobalTestArguments):
    haystack_path: str = field(
        default="long-llm:needle/PaulGrahamEssays",
        metadata={'help': 'The context for evaluation.'}
    )
    output_path: Optional[str] = field(
        default=None,
        metadata={'help': "The output file."}
    )
    test_length_min: Optional[int] = field(
        default=None,
        metadata={'help': "The minimum length of the input."}
    )
    test_length_max: Optional[int] = field(
        default=None,
        metadata={'help': "The maximum length of the input."}
    )
    test_length_num: Optional[int] = field(
        default=None,
        metadata={'help': "The number of the tested input lengths."}
    )
    test_length: List[int] = field(
        default_factory=lambda: [],
        metadata={'help': 'Specified evaluation lengths.'}
    )
    test_depth_min: Optional[int] = field(
        default=None,
        metadata={'help': "The minimum depth of the needle (from 0 to 100)."}
    )
    test_depth_max: Optional[int] = field(
        default=None,
        metadata={'help': "The maximum depth of the needle (from 0 to 100)."}
    )
    test_depth_num: Optional[int] = field(
        default=None,
        metadata={'help': "The number of the tested needle depth."}
    )
    test_depth: List[int] = field(
        default_factory=lambda: [],
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

    return inputs, context_return, needle


def needleTrain(context: str, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, **kwargs):
    context_dataset = ContextDataset(context, tokenizer, **kwargs)
    model = train(context_dataset, tokenizer, training_args, **kwargs)[0]
    return model


def main():
    (args, training_args, ttt_args), config = parse_args((TestArgs, TrainingArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)), no_dict=(TrainingArguments, TestArgs), return_config=True)

    tokenizer = load_tokenizer(ttt_args['model_name_or_path'])
    model_max_length = ttt_args['model_max_length']
    output_path = args.output_path
    
    # Needle in a haystack generation configs
    if len(args.test_depth) == 0:
        test_lengths = np.linspace(args.test_length_min, args.test_length_max, args.test_length_num, endpoint=True).astype(int).tolist()
    else:
        test_lengths = args.test_length
    if len(args.test_depth) == 0:
        test_depths = np.linspace(args.test_depth_min, args.test_depth_max, args.test_depth_num, endpoint=True).astype(int).tolist()
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
    pickle_name = os.path.join(args.haystack_path, f"{args.test_length_min}-{args.test_length_max}.pickle")
    if os.path.exists(pickle_name):
        print(f"Detect and load the cached input file {pickle_name}.")
        with open(pickle_name, "rb") as handle:
            all_inputs = pickle.load(handle)
    else:
        print(f"Find no cached input file.")
        all_inputs = []
        for length in tqdm(test_lengths, desc="Constructing Data"):
            for depth in test_depths:
                prompt, context, needle = generate_sample(
                    tokenizer=tokenizer,
                    context=context,
                    context_length=length, 
                    needle_depth=depth,
                    needle=args.needle,
                    prompt=args.prompt,
                    zh=args.zh
                )
                all_inputs.append(dict(
                    prompt=prompt,
                    context=context,
                    needle=needle,
                    length=length,
                    depth=depth
                ))
        print(f"Cache the input file in {pickle_name}")
        with open(pickle_name, 'wb') as handle:
            pickle.dump(all_inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Resume from the checkpoint
    if os.path.exists(output_path) and not args['overwrite']:
        print(f"Detect the output file {output_path}.")
        with open(output_path, 'r') as f:
            all_outputs = json.load(f)
        print(f"Resume {len(all_outputs) - 1} entries from {output_path}.")
    else:
        all_outputs = [config]
    # Forward
    num_samples = len(all_inputs)
    for sample_id, sample in enumerate(tqdm(all_inputs, desc="Evaluating")):
        if sample_id + 1 < len(all_outputs):
            continue
        prompt = sample['prompt']
        context = sample['context']
        time1 = time.time()
        model = needleTrain(context, tokenizer, training_args, **ttt_args)
        time2 = time.time()
        print(f"Sample {sample_id + 1} / {num_samples}: Training cost time = {time2 - time1}")
        with torch.no_grad():
            model.eval()
            messages = [
                {'role': 'system', 'content': "You are a helpful assistant."},
                {'role': 'user', 'content': prompt}
            ]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
            if input_ids.shape[-1] > model_max_length:
                input_ids = torch.concat((input_ids[:, :model_max_length//2], input_ids[-model_max_length//2:]), dim=-1)
            attention_mask = torch.ones_like(input_ids)
            output_ids = model.generate(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                max_new_tokens=50,
                num_beams=1,
                do_sample=False,
                temperature=1.,
                pad_token_id=tokenizer.pad_token_id,
            )[0]
        time3 = time.time()
        print(f"Sample {sample_id + 1} / {num_samples}: Generation cost time = {time3 - time2}")
        output_ids = output_ids[input_ids.shape[-1]:]
        pred = tokenizer.decode(output_ids, skip_special_tokens=True)

        all_outputs.append(dict(
            length=sample['length'],
            depth=sample['depth'],
            pred=pred,
            needle=sample['needle']
        ))
        with open(output_path, 'w') as f:
            json.dump(all_outputs, f, indent=4)
        time4 = time.time()
        print(f"Sample {sample_id + 1} / {num_samples}: Post-processing cost time = {time4 - time3}")


if __name__ == "__main__":
    main()

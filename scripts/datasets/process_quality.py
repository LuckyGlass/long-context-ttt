"""
We use quality-train as the development dataset to tune the hyperparameters.
Useful properties: title, article, set_unique_id
Info: 300 articles in total, sample 50 articles. QA generation follows the same method as synthetic QA generation.
"""
import json
import numpy as np
import torch
import tqdm
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
# Configurations
data_path = 'datasets/QuALITY/QuALITY.v1.0.1.htmlstripped.train'
np.random.seed(0)
num_events = 5
num_timeline_reorder = 2
number_article = 30
model_max_length = 8000


@torch.no_grad()
def timeline_reorder_gen(tokenizer, generator, full_context: str, num_events: int, model_max_length: 4096):
    def distribute_numbers(k, ell):
        split_points: list = np.random.choice(ell + 1, k - 1, replace=True).tolist()
        split_points.sort()
        split_points = [0] + split_points + [ell]
        return [r - l for l, r in zip(split_points[:-1], split_points[1:])]
    
    length_segments = 5  # Config
    sentences = sent_tokenize(full_context)
    length_gaps = distribute_numbers(num_events + 1, len(sentences) - num_events * length_segments)
    st_point = 0
    summaries = []
    for i in tqdm.tqdm(range(num_events), desc="Events"):
        st_point += length_gaps[i]
        context = ' '.join(sentences[st_point:st_point + length_segments])
        st_point += length_segments
        prompts = [
            "Please summary the event described in the following piece of texts in one sentence.",
            f"The piece of texts is: {context}",
            "Please summary the event described in the piece of texts in one sentence. Please do not output anything else."
        ]
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user', 'content': '\n'.join(prompts)}
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
        attention_masks = torch.ones_like(input_ids)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        output_ids = generator.generate(
            input_ids,
            max_new_tokens=1024,
            attention_mask=attention_masks,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=terminators,
            do_sample=False,
        )
        response = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        summaries.append(response)
    ranks = list(range(num_events))
    answers = list(range(num_events))
    np.random.shuffle(ranks)
    answers.sort(key=lambda i: ranks[i])
    prompts = [
        "Please sort the given events in the order of their appearance in the following long texts, from first to last.",
        full_context,
        "Please sort the given events in the order of their appearance in the long texts, from first to last. The given events are:",
    ]
    prompts += [f"[{i + 1}]: {summaries[j]}" for i, j in enumerate(ranks)]
    prompts += ["For example, a valid answer is [2] < [3] < [1] < [4] < [5]."]
    messages = [
        {'role': 'system', 'content': "You are a helpful assistant."},
        {'role': 'user', 'content': '\n'.join(prompts)},
        {'role': 'assistant', 'content': ' < '.join([f"[{i + 1}]" for i in answers])}
    ]
    return {'summaries': [summaries[i] for i in ranks], 'answers': answers}


def main():
    with open(data_path, 'r') as f:
        data = list(map(json.loads, f.readlines()))
    sampled_data = np.random.choice(data, size=number_article, replace=False)
    generator = AutoModelForCausalLM.from_pretrained('models/Meta-Llama-3-8B-Instruct')
    generator.eval()
    tokenizer = AutoTokenizer.from_pretrained('models/Meta-Llama-3-8B-Instruct')
    export_data = []
    for item in tqdm.tqdm(sampled_data, desc="Article"):
        export_data.append({
            'set_unique_id': item['set_unique_id'],
            'title': item['title'],
            'input': item['article'],
            'qa_pairs': [timeline_reorder_gen(tokenizer, generator, item['article'], num_events, model_max_length) for _ in tqdm.tqdm(range(num_timeline_reorder), desc="Samples")]
        })
        with open('datasets/QuALITY/dev.json', 'w') as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
    with open('datasets/QuALITY/dev.json', 'w') as f:
        json.dump(export_data, f, indent=4, ensure_ascii=False)


# fix some bugs about the evidences
def post_process():
    with open('datasets/QuALITY/dev.json', 'r') as f:
        data = json.load(f)
    for d in data:
        for qa_pair in d['qa_pairs']:
            evidence_text = qa_pair['S'][0]
            qa_pair['S'] = [s[2:] for s in evidence_text.split('\n')]
    with open('datasets/QuALITY/QuALITY_dev.json', 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()

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
    AutoTokenizer,
    BitsAndBytesConfig
)
# Configurations
model_name_or_path = 'models/Meta-Llama-3-8B-Instruct'
data_path = 'datasets/QuALITY/QuALITY.v1.0.1.htmlstripped.dev'
output_path = 'datasets/QuALITY/timeline-dev.json'
np.random.seed(0)
num_events = (3, 6)
num_timeline_reorder = 10
model_max_length = 8000


@torch.no_grad()
def timeline_reorder_gen(tokenizer, generator, full_context: str, num_events: int|tuple[int, int], model_max_length: 4096):
    def distribute_numbers(k, ell):
        split_points: list = np.random.choice(ell + 1, k - 1, replace=True).tolist()
        split_points.sort()
        split_points = [0] + split_points + [ell]
        return [r - l for l, r in zip(split_points[:-1], split_points[1:])]
    
    length_segments = 10  # Config
    sentences = sent_tokenize(full_context)
    if isinstance(num_events, tuple):
        num_events = np.random.randint(num_events[0], num_events[1] + 1)
    length_gaps = distribute_numbers(num_events + 1, len(sentences) - num_events * length_segments)
    st_point = 0
    summaries = []
    for i in range(num_events):
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
    return {
        'summaries': [summaries[i] for i in ranks],
        'answers': answers
    }


def main():
    with open(data_path, 'r') as f:
        data = list(map(json.loads, f.readlines()))
    generator = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
    )
    generator.eval()
    generator = torch.compile(generator)
    tokenizer = AutoTokenizer.from_pretrained('models/Meta-Llama-3-8B-Instruct')
    export_data = []
    for item in tqdm.tqdm(data, desc="Article"):
        export_data.append({
            'set_unique_id': item['set_unique_id'],
            'title': item['title'],
            'input': item['article'],
            'qa_pairs': [timeline_reorder_gen(tokenizer, generator, item['article'], num_events, model_max_length) for _ in tqdm.tqdm(range(num_timeline_reorder), desc="Samples")]
        })
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()

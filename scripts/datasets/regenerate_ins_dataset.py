"""
Generate datasets/train/ins_dataset.jsonl.
Force Llama to generate the original sentences as the evidences.
"""
#!/usr/bin/env python
# coding=utf-8

import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
import torch
import tqdm
import random
import re
import nltk
from nltk.tokenize import sent_tokenize


# Load the source datasets
files_source = [
    r'datasets/train/countingstar_ins.jsonl',
    r'datasets/train/infinitybench_ins.jsonl',
    r'datasets/train/lveval_ins.jsonl'
]

data_source = []
for file in files_source:
    with open(file, 'r') as f:
        data_source += list(map(json.loads, f.readlines()))

# Load the Llama-3 model and the tokenizer.
model_path = "models/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Generate QA pairs.
generated = []
for datapoint in tqdm.tqdm(data_source):
    article = datapoint["input"]
    sentences = sent_tokenize(article)
    num_sentences = len(sentences)
    count_failure = 0
    for i in range(0, num_sentences, 25):
        if num_sentences - i < 15:  # We require the context contains at least 15 sentences.
            break
        context = ' '.join(sentences[i:i + 25])
        messages = [
            {
                'role': "system",
                'content': "You are a helpful assistant."
            },
            {
                'role': "user", 
                'content': f"You are given a piece of text as the context. You should generate a question and the corresponding answer according to the context. You should also select one or more original sentences in the context as the evidences. Please answer in the following format:\nQuestion: [question]\nAnswer: [answer]\nEvidence:\n- [evidence 1]\n- [evidence 2]\n...\nPlease DON'T output quotes when outputing evidences. The following is the piece of text: {context}"
            }
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        mask_attention = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        num_of_trials = 0
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            attention_mask=mask_attention,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=terminators,
            do_sample=False,
        )
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        question_position = response.find("Question:")
        answer_position = response.find("Answer:")
        evidence_position = response.find("Evidence:")
        question = response[question_position + 9:answer_position].strip()
        answer = response[answer_position + 7:evidence_position].strip()
        evidences = response[evidence_position + 9:].strip().split('\n')
        evidences = list(map(lambda s: s[s.find('-') + 2:].strip(), evidences))
        if any(evidence.strip() not in context for evidence in evidences):  # FAILURE
            count_failure += 1
            continue
        pos_evidences = [context.find(e) for e in evidences]
        datapoint['qa_pairs'].append({
            'Q': question,
            'A': answer,
            'S': evidences,
            'Sp': pos_evidences
        })
    print(f"Generate {len(datapoint['qa_pairs'])} QA pairs. Fail {count_failure}")
    generated.append(datapoint)
    with open('datasets/train/ins_dataset_v2.json', 'w') as f:
        json.dump(generated, f, indent=4)

with open('datasets/train/ins_dataset_v2.json', 'w') as f:
    json.dump(generated, f, indent=4)

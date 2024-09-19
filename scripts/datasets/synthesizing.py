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

with open('datasets/train/lveval_ins.jsonl', 'r') as f:
    samples = f.readlines()[182:]
    
data = []
for sample in samples:
    data.append(json.loads(sample))

#llama3
model_path = "models/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

generated = []
for datapoint in tqdm.tqdm(data):
    text = datapoint["input"]
    texts = sent_tokenize(text)
    while len(texts) >= 15:
        if len(texts) >= 25:
            input_text = " ".join(texts[:25])
            texts = texts[25:]
        else:
            input_text = " ".join(texts)
            texts = []
        messages = [
            {
                'role': "system", 
                'content': "Given a piece of text, generate a question, an answer to the question and the evidence that supports the answer. The evidence should be the original sentences from the given text. Your answer should follow the format of the example below:\nQuestion: What is the capital of France?\nAnswer: Paris\nEvidence: Paris is the capital of France."
            },
            {
                'role': "user", 
                'content': f"Text: {input_text}"
            }
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        mask_attention = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        num_of_trials = 0
        while True:
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

            if question_position == -1 or answer_position == -1 or evidence_position == -1:
                num_of_trials += 1
                if num_of_trials > 5:
                    break
                continue
            else:
                question = response[question_position+9:answer_position].strip()
                answer = response[answer_position+7:evidence_position].strip()
                evidence = response[evidence_position+9:].strip()
                datapoint['qa_pairs'].append({"Q":question, "A":answer, "S":evidence})
                break
    generated.append(datapoint)
    with open('datasets/train/ins_dataset_v2.json', 'w') as f:
        json.dump(generated, f, indent=4)

with open('datasets/train/ins_dataset_v2.json', 'w') as f:
    json.dump(generated, f, indent=4)
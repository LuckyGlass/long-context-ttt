#!/usr/bin/env python
# coding=utf-8

import os
import json
import tqdm
import random
import re
from openai import OpenAI
import time
from nltk.tokenize import sent_tokenize

def get_title(text: str):
    client = OpenAI(
        api_key=os.getenv('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-max-longcontext",
        messages=[
            {'role': 'system', 'content': "Given a passage, generate a title. Your answer should only contain the title."},
            {'role': 'user', 'content': f"The passage is: {text}"},],
        temperature=0.8,
        top_p=0.8
    )
    response = completion.model_dump_json()
    response = json.loads(response)
    return response['choices'][0]['message']['content']

def get_question(text: str, qa_pairs: list):
    client = OpenAI(
        api_key=os.getenv('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    cnt = 0
    while len(qa_pairs) < 10:
        prompt = "Given some texts, generate a question, the answer to the question and the evidence that supports the answer. The evidence should be original sentences from the given texts. Your answer should follow the format of the example below, but your question should not be similar to the example:\n"
        qa_pair = qa_pairs[0]
        prompt += f"Question: {qa_pair['Q']}\nAnswer: {qa_pair['A']}\nEvidence: {qa_pair['S']}"
        try:
            completion = client.chat.completions.create(
                model="qwen-max",
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': f"The texts are:{text[cnt*1000:(cnt+1)*1000]}"},],
                temperature=0.8,
                top_p=0.8
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(60)
            continue
        
        response = completion.model_dump_json()
        response = json.loads(response)
        response = response['choices'][0]['message']['content']
        #print(response)
        
        question_position = response.find("Question:")
        answer_position = response.find("Answer:")
        evidence_position = response.find("Evidence:")

        if question_position == -1 or answer_position == -1 or evidence_position == -1:
            continue
        else:
            question = response[question_position+9:answer_position].strip()
            answer = response[answer_position+7:evidence_position].strip()
            evidence = response[evidence_position+9:].strip()
            qa_pairs.append({"Q":question, "A":answer, "S":evidence})
            cnt+=1
            #print(cnt)
    
    return qa_pairs

with open('datasets/bamboo/paperqa_16k.jsonl', 'r') as f:
    lines = f.readlines()
    samples = []
    for line in lines:
        try:
            samples.append(json.loads(line))
        except:
            continue

results = []
for sample in tqdm.tqdm(samples[:60]):
    new_sample = {}
    new_sample["input"] = sample["content"]
    new_sample["qa_pairs"] = []
    answer = sample["answer"]
    options = sample["options"]
    for option in options:
        if option[0] == answer:
            answer = option[3:]
            break
    new_sample["qa_pairs"].append({"Q": sample["question"], "A": answer, "S": sample["evidence"]})
    
    # generate title
    new_sample["title"] = get_title(sample["content"])
    # generate question
    new_sample["qa_pairs"] = get_question(sample["content"], new_sample["qa_pairs"])
    
    results.append(new_sample)
    with open('datasets/bamboo/ins_dataset2.json', 'w') as f:
        json.dump(results, f, indent=4)
with open('datasets/bamboo/ins_dataset2.json', 'w') as f:
    json.dump(results, f, indent=4)
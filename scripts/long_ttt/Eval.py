#!/usr/bin/env python
# coding=utf-8
import tqdm
import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
import numpy as np
from bert_score import BERTScorer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
import torch
import re
import itertools
import roman
from datasets import load_dataset
import copy
import openai, os

# for debugging
class Data_args:
    def __init__(self):
        self.len_provide = 10
        self.prepend_title = True
class Training_args:
    def __init__(self):
        self.device = 'cuda'
class Output_args:
    def __init__(self):
        self.prepend_input = True
        self.recite_first = False
class Model_args:
    def __init__(self):
        self.model_max_length = 8000


# evaluate on long_qa or short_qa with bertscore and meteor. Perlexity and retrieval are also supported as supplementary metrics to these two tasks
class Eval:
    def __init__(self, metrics: list):
        """
        Args:
            metrics (list): options[long_qa, perplexity, short_qa, retrieval]
        """
        if 'long_qa' not in metrics and 'short_qa' not in metrics:
            raise NotImplementedError("You must choose whether to evaluate on long_qa or short_qa.")
        if 'long_qa' in metrics and 'short_qa' in metrics:
            raise NotImplementedError("You can only choose one of long_qa and short_qa.")
        
        self.metrics = metrics
        self.results = {}
        if 'long_qa' in metrics:
            self.results['long_qa'] = {'meteor':[], 'bertscore':[]}
        if 'short_qa' in metrics:
            self.results['short_qa'] = {'meteor':[], 'bertscore':[]}
        if 'perplexity' in metrics:
            self.results['perplexity'] = []
        if 'retrieval' in metrics:
            self.results['meteor_recall'] = []
            
        if 'long_qa' in metrics or 'short_qa' in metrics:
            # bertscorer
            model_path = 'models/roberta-large'
            self.bert_scorer = BERTScorer(model_type=model_path, device="cuda:0")
    
    def get_bleu_score(self, reference: list, pred: str):
        reference = [ref.replace("\n", " ").split() for ref in reference]
        pred = pred.replace("\n", " ").split()
        bleu_score = sentence_bleu(reference, pred, weights=(0.25, 0.25, 0.25, 0.25))
        return bleu_score

    def get_meteor_score(self, reference: str, pred: str):
        reference, pred = (
            reference.replace("\n", " ").split(),
            pred.replace("\n", " ").split(),
        )
        meteor = single_meteor_score(reference, pred)
        return meteor
    
    def get_bertscore(self, reference: str, pred: str):
        P, R, F1 = self.bert_scorer.score([pred], [reference], verbose=False)
        return F1.item()
        
    # recite
    def get_perplexity(self, evidence: str, model, tokenizer, training_args, data_args):
        encodings = tokenizer(evidence, return_tensors="pt")
        input_ids = encodings.input_ids.to(training_args.device)
    
        len_origin = input_ids.shape[1]
        
        if len_origin <= data_args.len_provide:
            return None
    
        target_ids = encodings.input_ids.clone()
        target_ids[:, :(data_args.len_provide-1)] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        ppl = torch.exp(neg_log_likelihood)
        return ppl.item()
    
    
    def evaluate_qa(self, question: str, reference: str, pred: str):
        
        scores = {}
        scores['meteor'] = self.get_meteor_score(reference, pred)
        scores['bertscore'] = self.get_bertscore(reference, pred)

        return scores
    
    def test_retrieve(self, title: str, qa_pair, model, tokenizer, data_args, training_args):
        question = qa_pair['Q']
        if type(qa_pair['S']) == str:
            reference = [qa_pair['S']]
        else:
            reference = qa_pair['S']
        if data_args.prepend_title:
            messages = [
                {'role': "system", 'content': "You are a helpful assistant. "},
                {'role': "user", 'content': f"Please recite the evidences from the text \"{title}\" for answering the following question. Please use '####' to separate each of the evidences.\nQuestion: {question}\nAnswer:"}
            ]
        else:
            messages = [
                {'role': "system", 'content': "You are a helpful assistant."},
                {'role': "user", 'content': f"Please recite the evidences for answering the following question. Please use '####' to separate each of the evidences.\nQuestion:{question}\nAnswer:"}
            ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        mask_attention = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
        output = model.generate(
            input_ids,
            max_new_tokens=1024,
            attention_mask=mask_attention,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        output_model = output[0][input_ids.shape[-1]:]
        pred = tokenizer.decode(output_model, skip_special_tokens=True)
        qa_pair['S_recite'] = pred
        
        # meteor
        pred = pred.split("####")
        meteor_recall = []
        for r in reference:
            flag = 0
            for p in pred:
                if self.get_meteor_score(r, p) > 0.5:
                    flag = 1
                    break
            meteor_recall.append(flag)
        meteor_recall = sum(meteor_recall)/len(meteor_recall)
        
        return meteor_recall
    
    def evaluate_with_evidence(self, datapoint, model, tokenizer, training_args, data_args, output_args, model_args):
        """
        Return:
            **`datapoint` is modified but not returned**
        """
        if type(datapoint['qa_pairs']) == str:
            datapoint['qa_pairs'] = eval(datapoint['qa_pairs'])
        if 'long_qa' in self.metrics:
            for qa_pair in tqdm.tqdm(datapoint['qa_pairs']):
                prompts = []
                prompts += [
                    "Please answer the question based on the following facts:"
                ]
                prompts += [s for s in qa_pair['S']]
                prompts += [
                    f"Question: {qa_pair['Q']}",
                ]
                messages = [
                    {'role': 'system', 'content': "You are a helpful assistant. "},
                    {'role': 'user', 'content': '\n'.join(prompts)}
                ]
                input_ids = torch.LongTensor(tokenizer.apply_chat_template(messages, add_generation_prompt=True))[None, :].to(model.device)
                if output_args.prepend_input:
                    max_length = model_args.model_max_length
                    input_ids = torch.cat((input_ids[:, :max_length//2], input_ids[:, -max_length//2:]), dim=1)
                mask_attention = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
                terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                output = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    attention_mask=mask_attention,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=terminators,
                    do_sample=False,
                    temperature=1.0,
                    num_beams=1,
                    repetition_penalty=1.1,
                )
                output_model = output[0][input_ids.shape[-1]:]
                pred = tokenizer.decode(output_model, skip_special_tokens=True)
                scores = self.evaluate_qa(qa_pair['Q'], qa_pair['A'], pred)
                for metric_name, score in scores.items():
                    self.results['long_qa'][metric_name].append(score)
                qa_pair['pred'] = pred
                qa_pair['scores'] = scores
                
        if 'short_qa' in self.metrics:
            for qa_pair in tqdm.tqdm(datapoint['qa_pairs']):
                prompts = []
                prompts += [
                    "Please answer the question based on the following fact:"
                ]
                prompts += [qa_pair['S']]
                prompts += [
                    f"Question: {qa_pair['Q']}",
                ]
                messages = [
                    {'role': 'system', 'content': "You are a helpful assistant. "},
                    {'role': 'user', 'content': '\n'.join(prompts)}
                ]
                input_ids = torch.LongTensor(tokenizer.apply_chat_template(messages, add_generation_prompt=True))[None, :].to(model.device)
                if output_args.prepend_input:
                    max_length = model_args.model_max_length
                    input_ids = torch.cat((input_ids[:, :max_length//2], input_ids[:, -max_length//2:]), dim=1)
                mask_attention = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
                terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                output = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    attention_mask=mask_attention,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=terminators,
                    do_sample=False,
                    temperature=1.0,
                    num_beams=1,
                    repetition_penalty=1.1,
                )
                output_model = output[0][input_ids.shape[-1]:]
                pred = tokenizer.decode(output_model, skip_special_tokens=True)
                scores = self.evaluate_qa(qa_pair['Q'], qa_pair['A'], pred)
                
                for metric_name, score in scores.items():
                    self.results['short_qa'][metric_name].append(score)
                qa_pair['pred'] = pred
                qa_pair['scores'] = scores
                    
                
        
    def evaluate_datapoint(self, datapoint, model, tokenizer, training_args, data_args, output_args, model_args):
        """
        Return:
            **`datapoint` is modified but not returned**
        """
        if type(datapoint['qa_pairs']) == str:
            datapoint['qa_pairs'] = eval(datapoint['qa_pairs'])
        # long qa & perplexity & retrieval
        if 'long_qa' in self.metrics:
            for qa_pair in tqdm.tqdm(datapoint['qa_pairs']):
                # long qa
                prompts = []
                if data_args.prepend_title:
                    prompts += [
                        f"Please answer the following question only based on \"{datapoint['title']}\"."
                    ]
                if output_args.prepend_input:
                    prompts += [
                        f"This is part of the texts from \"{datapoint['title']}\": \"{datapoint['input']}\""
                    ]
                if output_args.recite_first:
                    prompts += [
                        f"Please recite the facts from \"{datapoint['title']}\" that support your answer before answering the question according to the facts.",
                        f"Question: {qa_pair['Q']}",
                        f"Please answer in the following format: \"Evidence: <facts>. Answer: <answer>\". Do not output anything else.",
                    ]
                else:
                    prompts += [
                        f"Question: {qa_pair['Q']}",
                    ]
                    
                messages = [
                    {'role': 'system', 'content': "You are a helpful assistant. "},
                    {'role': 'user', 'content': '\n'.join(prompts)}
                ]
                input_ids = torch.LongTensor(tokenizer.apply_chat_template(messages, add_generation_prompt=True))[None, :].to(model.device)
                if output_args.prepend_input:
                    max_length = model_args.model_max_length
                    input_ids = torch.cat((input_ids[:, :max_length//2], input_ids[:, -max_length//2:]), dim=1)
                mask_attention = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
                terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                output = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    attention_mask=mask_attention,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=terminators,
                    do_sample=False,
                    temperature=1.0,
                    num_beams=1,
                    repetition_penalty=1.1,
                )
                output_model = output[0][input_ids.shape[-1]:]
                pred = tokenizer.decode(output_model, skip_special_tokens=True)
                scores = self.evaluate_qa(qa_pair['Q'], qa_pair['A'], pred)
                for metric_name, score in scores.items():
                    self.results['long_qa'][metric_name].append(score)
                qa_pair['pred'] = pred
                qa_pair['scores'] = scores
                        
                # perplexity
                if 'perplexity' in self.metrics:
                    ppl = []
                    for evidence in qa_pair["S"]:
                        perplexity = self.get_perplexity(evidence, model, tokenizer, training_args, data_args)
                        if perplexity:
                            ppl.append(perplexity)
                            self.results['perplexity'].append(perplexity)
                    if len(ppl) > 0:
                        ppl = sum(ppl)/len(ppl)
                    else:
                        ppl = 'None'
                    qa_pair['perplexity'] = ppl
                
                # retrieval
                if 'retrieval' in self.metrics:
                    recall = self.test_retrieve(datapoint['title'], qa_pair, model, tokenizer, data_args, training_args)
                    qa_pair['meteor_recall'] = recall
                    self.results['meteor_recall'].append(recall)
        
        # short qa
        if 'short_qa' in self.metrics:
            for qa_pair in tqdm.tqdm(datapoint['qa_pairs']):
                prompts = []
                if data_args.prepend_title:
                    prompts += [
                        f"Please answer the following question only based on \"{datapoint['title']}\"."
                    ]
                if output_args.prepend_input:
                    prompts += [
                        f"This is part of the texts from \"{datapoint['title']}\": \"{datapoint['input']}\""
                    ]
                if output_args.recite_first:
                    prompts += [
                        f"Please recite the facts from \"{datapoint['title']}\" that support your answer before answering the question according to the facts.",
                        f"Question: {qa_pair['Q']}",
                        f"Please answer in the following format: \"Evidence: <facts>. Answer: <answer>\". Do not output anything else.",
                    ]
                else:
                    prompts += [
                        f"Question: {qa_pair['Q']}",
                    ]
                messages = [
                    {'role': 'system', 'content': "You are a helpful assistant. "},
                    {'role': 'user', 'content': '\n'.join(prompts)}
                ]
                input_ids = torch.LongTensor(tokenizer.apply_chat_template(messages, add_generation_prompt=True))[None, :].to(model.device)
                if output_args.prepend_input:
                    max_length = model_args.model_max_length
                    input_ids = torch.cat((input_ids[:, :max_length//2], input_ids[:, -max_length//2:]), dim=1)
                mask_attention = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
                terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                output = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    attention_mask=mask_attention,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=terminators,
                    do_sample=False,
                    temperature=1.0,
                    num_beams=1,
                    repetition_penalty=1.1,
                )
                output_model = output[0][input_ids.shape[-1]:]
                pred = tokenizer.decode(output_model, skip_special_tokens=True)
                scores = self.evaluate_qa(qa_pair['Q'], qa_pair['A'], pred)
                
                for metric_name, score in scores.items():
                    self.results['short_qa'][metric_name].append(score)
                qa_pair['pred'] = pred
                qa_pair['scores'] = scores
                        
                # perplexity
                if 'perplexity' in self.metrics:
                    perplexity = self.get_perplexity(qa_pair["S"], model, tokenizer, training_args, data_args)
                    if perplexity:
                        self.results['perplexity'].append(perplexity)
                    qa_pair['perplexity'] = perplexity
                        
                # retrieval
                if 'retrieval' in self.metrics:
                    recall = self.test_retrieve(datapoint['title'], qa_pair, model, tokenizer, data_args, training_args)
                    qa_pair['meteor_recall'] = recall
                    self.results['meteor_recall'].append(recall)
            
    # sum the scores saved in results
    def get_final_results(self):
        tmp_results = copy.deepcopy(self.results)
        for key, value in tmp_results.items():
            if key == 'long_qa' or key == 'short_qa':
                for metric_name, scores in value.items():
                    if len(scores) > 0:
                        tmp_results[key][metric_name] = sum(scores)/len(scores)
            else:
                if len(value) > 0:
                    tmp_results[key] = sum(value)/len(value)
        return tmp_results
    
    # clear the results
    def clear_results(self):
        self.results = {}
        if 'long_qa' in self.metrics:
            self.results['long_qa'] = {'meteor':[], 'bertscore':[]}
        if 'short_qa' in self.metrics:
            self.results['short_qa'] = {'meteor':[], 'bertscore':[]}
        if 'perplexity' in self.metrics:
            self.results['perplexity'] = []
        if 'retrieval' in self.metrics:
            self.results['meteor_recall'] = []
    
    
if __name__ == "__main__":
    
    # for debugging
    model_path = "models/Meta-Llama-3-8B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    data_args = Data_args()
    training_args = Training_args()
    output_args = Output_args()
    Model_args = Model_args()
    
    with open("datasets/loogle/shortdep_qa.json", "r") as f:
        raw_datasets = f.readlines()
    
    metrics = ['short_qa', 'perplexity']
    
    evaluator = Eval(metrics=metrics)
    
    results = []
    for data in tqdm.tqdm(raw_datasets[:1]):
        datapoint = eval(data)
        datapoint['qa_pairs'] = eval(datapoint['qa_pairs']) 
        evaluator.evaluate_datapoint(datapoint, model, tokenizer, training_args, data_args, output_args, Model_args)
        results.append(datapoint)
        torch.cuda.empty_cache()
        
    scores = evaluator.get_final_results()
    results.insert(0, scores)
    
    with open("output/debug.json", "w") as f:
        json.dump(results, f, indent=4)

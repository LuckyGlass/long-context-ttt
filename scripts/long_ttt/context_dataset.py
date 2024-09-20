from nltk.tokenize import sent_tokenize
import torch
from copy import deepcopy
from torch.utils.data import Dataset
import transformers
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM
)
from typing import Optional

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

def apply_qa_template(tokenizer: transformers.PreTrainedTokenizer, question: str, answer: str, evidences: list[str], title: str, prepend_title: bool=False, sent_token: str|None=None) -> tuple[torch.Tensor, int]:
    """Apply the QA template, used for training.
    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer; it should be equipped with a chat template.
        question (str):
        answer (str):
        evidences (list[str]): the evidences; it should be presented as a list of sentences.
        title (str):
        prepend_title (bool): OPTIONAL; whether to prompt the model with the title.
        sent_token (str|None): if specified as non-None value, a `sent_token` will be prepended to each sentence in the evidences.
    Returns:
        PAIR (tuple[Tensor, int]): input_ids - the `input_ids` of the model; len_input - the length of the unsupervised texts, including the system prompt, the context, and the question.
    """
    if sent_token is None:
        text_evidences = ' '.join(evidences)
    else:
        text_evidences = sent_token.join(evidences)
    if prepend_title:
        messages = [
            {'role': "system", 'content': "You are a helpful assistant. "},
            {'role': "user", 'content': f"Please answer the following question only based on \"{title}\".\nQuestion: {question}"},
            {'role': "assistant", 'content': f"The related facts are: {text_evidences}\nSo the answer should be: {answer}"}
        ]
    else:
        messages = [
            {'role': "system", 'content': "You are a helpful assistant."},
            {'role': "user", 'content': f"Please answer the following question.\nQuestion: {question}"},
            {'role': "assistant", 'content': f"The related facts are: {text_evidences}\nSo the answer should be: {answer}"}
        ]
    len_input = len(tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True))
    return tokenizer.apply_chat_template(messages, add_generation_prompt=False), len_input


class ContextDataset(Dataset):
    shared_generator: Optional[PreTrainedModel] = None
    
    def __init__(self, context: str, tokenizer: transformers.PreTrainedTokenizer, title: Optional[str]=None, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, prepend_title: bool=False, sent_token: bool=False, num_generate_qa: int=0, generator_name_or_path: Optional[str]=None, **kwargs):
        """
        Args:
            context (str): the context to train on.
            tokenizer (PreTrainedTokenizer): the AutoTokenizer.
            model_max_length (int): OPTIONAL, default to `4096`; the texts will be clipped at the `model_max_length`-th token.
            block_size (int): OPTIONAL, default to `256`; the number of tokens in a block; a block is the unit of segments and offsets.
            len_segment (int): OPTIONAL, default to `8`; the number of units in a segment; the article is divided into segments.
            len_offset (int): OPTIONAL, default to `3`; the number of units per offset; it determines the offset from one segment to the next one.
            prepend_title (bool): OPTIONAL, default to `False`; whether to prompt the model with the title.
            sent_token (bool): OPTIONAL, default to `False`; whether to insert a `<|reserved_special_token_249|>` between each two sentences; if enabled, the model must be trained to recognize this token.
        """
        self.ignore_index = -100  # The default value for ignored labels in torch
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.sent_token = '<|reserved_special_token_249|>' if sent_token else None
        texts = context.replace('\0', ' ')
        if sent_token:
            sentences = sent_tokenize(texts)
            texts = self.sent_token.join(sentences)
        if prepend_title and title is not None:
            texts = f"Title: {title}.\nContent: {texts}"
        texts = self.tokenizer.bos_token + texts + self.tokenizer.eos_token  # Manually add special tokens
        input_ids = self.tokenizer(texts, add_special_tokens=False)['input_ids']
        len_segment = len_segment * block_size
        len_offset = len_offset * block_size
        # Generate datapoints
        self.data = [input_ids[s:s+len_segment] for s in range(0, len(input_ids), len_offset)]
        # Generate QA
        if self.shared_generator is None and generator_name_or_path is not None:
            self.shared_generator = AutoModelForCausalLM.from_pretrained(
                generator_name_or_path,
                trust_remote_code=True,
                device_map="auto",
            )
            self.shared_generator.eval()
        if num_generate_qa > 0:
            short_qas = self.shortqa_gen(context, num_generate_qa)
        # TODO: apply the qa template
    
    def shortqa_gen(self, context: str, num_generate_qa: int=0):
        generated = []
        texts = sent_tokenize(context)
        print(len(texts))
        c = 0
        while len(texts) >= 15:
            if len(texts) >= 25:
                context = " ".join(texts[:25])
                texts = texts[25:]
            else:
                context = " ".join(texts)
                texts = []
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
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.shared_generator.device)
            mask_attention = torch.ones(input_ids.shape, dtype=torch.long, device=self.shared_generator.device)
            terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            num_of_trials = 0
            while True:
                outputs = self.shared_generator.generate(
                    input_ids,
                    max_new_tokens=1024,
                    attention_mask=mask_attention,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=terminators,
                    do_sample=False,
                )
                response = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
                print(response)
                question_position = response.find("Question:")
                answer_position = response.find("Answer:")
                evidence_position = response.find("Evidence:")
                question = response[question_position + 9:answer_position].strip()
                answer = response[answer_position + 7:evidence_position].strip()
                evidences = response[evidence_position + 9:].strip().split('\n')
                evidences = list(map(lambda s: s[s.find('-') + 2:].strip(), evidences))
                c += 1
                if question_position == -1 or answer_position == -1 or evidence_position == -1:
                    num_of_trials += 1
                    if num_of_trials > 5:
                        break
                    continue
                else:
                    print('=====================================')
                    question = response[question_position+9:answer_position].strip()
                    answer = response[answer_position+7:evidence_position].strip()
                    evidence = response[evidence_position+9:].strip()
                    generated.append({"Q":question, "A":answer, "S":evidence})
                    break

        if len(generated) > num_generate_qa:
            generated = generated[:num_generate_qa+1]

        print(generated, len(generated))
        return generated
    
    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        if isinstance(example, tuple):
            example, len_input = example
        else:
            len_input = 0
        input_ids = example
        labels = example
        # Clip and truncation
        input_ids = input_ids[:self.model_max_length]
        labels = labels[:self.model_max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))
        labels += [self.ignore_index] * (self.model_max_length - len(labels))
        # Transfer to Tensor
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        labels[:len_input] = self.ignore_index  # mask the unsupervised part
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    def __getitem__(self, index):
        return self.preprocessing(self.data[index])

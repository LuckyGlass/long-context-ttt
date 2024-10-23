from nltk.tokenize import sent_tokenize
import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
from typing import Optional
from numpy.random import randint, choice, shuffle
import tqdm


PRE_PROMPT = "Given a long text, reorder the events according to the original order of the events in the long text."
POST_PROMPT = "Events: {events}\n\nGive the reorder results."


def apply_qa_template(question: str, answer: Optional[str]=None, evidences: list[str]=[], title: Optional[str]=None, context: Optional[str]=None, prepend_title: bool=False, sent_token: Optional[str]=None, recite_first: bool=False, enable_ICL: bool=False, return_answer: bool=False):
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
        sentences = sent_tokenize(context)
        context = sent_token.join(sentences)
        text_evidences = sent_token.join(evidences)
    prompts = []
    if prepend_title:
        prompts.append(f"Please answer the following question only based on \"{title}\".")
        if enable_ICL:
            prompts.append(f"This is part of the texts from \"{title}\": \"{context}\"")
        if recite_first:
            prompts.append(f"Please recite the facts from \"{title}\" that support your answer before answering the question according to the facts.")
    else:
        prompts.append("Please answer the following question.")
        if enable_ICL:
            prompts.append(f"This is part of the texts: \"{context}\"")
        if recite_first:
            prompts.append(f"Please recite the facts from the text that support your answer before answering the question according to the facts.")
    prompts.append(f"Question: {question}")
    if recite_first:
        prompts.append("Please answer in the following format: \"Evidence: <facts>. Answer: <answer>\". Do not output anything else.")
    if return_answer:
        if recite_first:
            output = f"Evidence: {text_evidences} Answer: {answer}"
        else:
            output = answer
        return '\n'.join(prompts), output
    else:
        return '\n'.join(prompts)


class ContextDataset(Dataset):
    def __init__(self, context: str, tokenizer: transformers.PreTrainedTokenizer, title: Optional[str]=None, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, prepend_title: bool=False, sent_token: bool=False, num_generate_qa: int=0, generator_name_or_path: Optional[str]=None, qa_loss_weight: float=1.0, pad_to_max_length: bool=True, ttt_recite_first: bool=False, ttt_enable_ICL: bool=False, involve_qa_epochs: int=0, enable_diverse_qa: bool=False, num_timeline_reorder: int=0, num_timeline_reorder_events: int|tuple[int, int]=5, append_question: bool=False, **kwargs):
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
        self.pad_to_max_length = pad_to_max_length
        self.qa_loss_weight = qa_loss_weight
        texts = context.replace('\0', ' ')
        if sent_token:
            sentences = sent_tokenize(texts)
            texts = self.sent_token.join(sentences)
        if prepend_title and title is not None:
            texts = f"Title: {title}.\nContent: {texts}"
        input_ids = self.tokenizer(texts, add_special_tokens=False)['input_ids']
        len_segment = len_segment * block_size
        len_offset = len_offset * block_size
        # Generate datapoints
        if append_question:
            assert 'events' in kwargs, "append_question=True but no events provided."
            events = kwargs.pop('events')
            prepend_ids = self.tokenizer(PRE_PROMPT, add_special_tokens=False)['input_ids']
            question_ids = self.tokenizer(POST_PROMPT.format_map({'events': events}), add_special_tokens=False)['input_ids']
            self.context_data = [prepend_ids + input_ids[s:s+len_segment] + question_ids for s in range(0, len(input_ids), len_offset)]
        else:
            self.context_data = [input_ids[s:s+len_segment] for s in range(0, len(input_ids), len_offset)]
        self.num_segments = len(self.context_data)  # record the number of context datapoints
        # Generate QA
        self.num_generate_qa = num_generate_qa
        self.enable_diverse_qa = enable_diverse_qa
        self.counter_qa = 0
        self.counter_timline_reorder = 0
        if enable_diverse_qa:
            num_generate_qa *= involve_qa_epochs
        generator = None
        if generator_name_or_path is not None:
            generator = AutoModelForCausalLM.from_pretrained(
                generator_name_or_path,
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
        self.qa_data = []
        if num_generate_qa > 0:
            temp_qas = self.shortqa_gen(generator, context, num_generate_qa)
            torch.cuda.empty_cache()
            for qa in temp_qas:
                user_msg, assistant_msg = apply_qa_template(
                    question=qa['Q'],
                    answer=qa['A'],
                    evidences=qa['S'],
                    title=title,
                    context=context,
                    prepend_title=prepend_title,
                    sent_token=self.sent_token,
                    recite_first=ttt_recite_first,
                    enable_ICL=ttt_enable_ICL,
                    return_answer=True
                )
                messages = [
                    {'role': 'system', 'content': "You are a helpful assistant."},
                    {'role': 'user', 'content': user_msg},
                    {'role': 'assistant', 'content': assistant_msg}
                ]
                input_length = len(self.tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True))
                input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False)
                output_length = len(input_ids) - input_length
                if len(input_ids) > model_max_length:
                    input_ids = input_ids[:model_max_length//2] + input_ids[-model_max_length//2:]
                    input_length = len(input_ids) - output_length
                self.qa_data.append((input_ids, input_length))
        # Generate timeline reorder
        self.reorder_data = []
        self.num_timeline_reorder = num_timeline_reorder
        if enable_diverse_qa:
            num_timeline_reorder *= involve_qa_epochs
        for _ in tqdm.tqdm(range(num_timeline_reorder), desc="Generate timeline reorder tasks"):
            self.reorder_data.append(self.timeline_reorder_gen(generator, context, num_timeline_reorder_events, model_max_length))
        # Add a tag - whether to involve synthetic QA pairs
        del generator
        self.involve_qa = False
    
    def enable_qa(self):
        self.involve_qa = True
    
    def disable_qa(self):
        self.involve_qa = False
    
    @torch.no_grad()
    def shortqa_gen(self, generator, full_context: str, num_generate_qa: int=0):
        generated = []
        texts = sent_tokenize(full_context)
        c = 0
        assert len(texts) >= 25
        for _ in tqdm.tqdm(range(num_generate_qa), desc="Generate QA"):
            st_pos = randint(0, len(texts) - 25)
            context = ' '.join(texts[st_pos:st_pos+25])
            messages = [
                {
                    'role': "system",
                    'content': "You are a helpful assistant."
                },
                {
                    'role': "user", 
                    'content': f"You are given a piece of text as the context. You should generate ONLY one question and the corresponding answer according to the context. You should also select one or more original sentences in the context as the evidences. Please answer in the following format:\nQuestion: [question]\nAnswer: [answer]\nEvidence:\n- [evidence 1]\n- [evidence 2]\n...\nPlease DON'T output quotes when outputing evidences. The following is the piece of text: {context}"
                }
            ]
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            mask_attention = torch.ones_like(input_ids)
            terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            num_of_trials = 0
            while True:
                outputs = generator.generate(
                    input_ids,
                    max_new_tokens=1024,
                    attention_mask=mask_attention,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=terminators,
                    do_sample=False,
                )
                response = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
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
                    question = response[question_position+9:answer_position].strip()
                    answer = response[answer_position+7:evidence_position].strip()
                    evidence = response[evidence_position+9:].strip()
                    generated.append({"Q":question, "A":answer, "S":evidence})
                    break
        return generated
    
    @torch.no_grad()
    def timeline_reorder_gen(self, generator, full_context: str, num_events: int|tuple[int, int], model_max_length: 4096):
        def distribute_numbers(k, ell):
            split_points: list = choice(ell + 1, k - 1, replace=True).tolist()
            split_points.sort()
            split_points = [0] + split_points + [ell]
            return [r - l for l, r in zip(split_points[:-1], split_points[1:])]
        
        length_segments = 10  # Config
        sentences = sent_tokenize(full_context)
        if isinstance(num_events, tuple):
            num_events = randint(num_events[0], num_events[1] + 1)
        length_gaps = distribute_numbers(num_events + 1, len(sentences) - num_events * length_segments)
        st_point = 0
        summaries = []
        for i in range(num_events):
            st_point += length_gaps[i]
            context = ' '.join(sentences[st_point:st_point + length_segments])
            st_point += length_segments
            prompts = [
                "Please summary the event described in the following piece of texts in 2 sentences.",
                f"The piece of texts is: {context}",
                "Please summary the event described in the piece of texts in 2 sentences. Please do not output anything else."
            ]
            messages = [
                {'role': 'system', 'content': "You are a helpful assistant."},
                {'role': 'user', 'content': '\n'.join(prompts)}
            ]
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
            attention_masks = torch.ones_like(input_ids)
            terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            output_ids = generator.generate(
                input_ids,
                max_new_tokens=1024,
                attention_mask=attention_masks,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=terminators,
                do_sample=False,
            )
            response = self.tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
            summaries.append(response)
        ranks = list(range(num_events))
        answers = list(range(num_events))
        shuffle(ranks)
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
        input_length = len(self.tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True))
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        output_length = len(input_ids) - input_length
        if len(input_ids) > model_max_length:
            input_ids = input_ids[:model_max_length//2] + input_ids[-model_max_length//2:]
            input_length = len(input_ids) - output_length
        return (input_ids, input_length)
    
    def __len__(self):
        # Modify __len__ to decide whether to involve synthetic QA pairs
        if self.involve_qa:
            return self.num_segments + self.num_generate_qa + self.num_timeline_reorder
        else:
            return self.num_segments  # the first several datapoints are the context

    def preprocessing(self, example, index):
        if isinstance(example, tuple):
            example, len_input = example
        else:
            len_input = 0
        input_ids = example
        labels = example
        # Clip and truncation
        input_ids = input_ids[:self.model_max_length]
        labels = labels[:self.model_max_length]
        if self.pad_to_max_length:
            input_ids += [self.tokenizer.eos_token_id] * (self.model_max_length - len(input_ids))
            labels += [self.ignore_index] * (self.model_max_length - len(labels))
        # Transfer to Tensor
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        labels[:len_input] = self.ignore_index  # mask the unsupervised part
        attention_mask = input_ids.ne(self.tokenizer.eos_token_id)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'weights': 1.0 if index < self.num_segments else self.qa_loss_weight
        }
    
    def __getitem__(self, index):
        if index < self.num_segments:
            return self.preprocessing(self.context_data[index], index)
        elif index < self.num_generate_qa + self.num_segments:
            if self.counter_qa == len(self.qa_data):
                self.counter_qa = 0
            self.counter_qa += 1
            return self.preprocessing(self.qa_data[self.counter_qa - 1], index)
        elif index < self.num_timeline_reorder + self.num_generate_qa + self.num_segments:
            if self.counter_timline_reorder == len(self.reorder_data):
                self.counter_timline_reorder = 0
            self.counter_timline_reorder += 1
            return self.preprocessing(self.reorder_data[self.counter_timline_reorder - 1], index)
        else:
            raise ValueError(f'Index {index} too large.')

import argparse
import base64
import csv
import io
import json
import logging
import os
import pickle
import tiktoken
import time
import tqdm
from collections import defaultdict
from nltk import sent_tokenize
from openai import OpenAI
from gpt import FTDataset, FTGPT, dump_data
import re

logging.basicConfig(
    filename='logs/gpt3.5_assis_only_b2048_o1024_lrm0.5_epoch3_bamboo_eval.log',
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    filemode='w'
)

client = OpenAI(api_key=os.getenv('GPT3_API_KEY'))

# for reportsumsort
def get_prediction(fine_tuned_model: str, sample: str, max_input_length: int):
    prompt = {
        "system": "You are a helpful assistant.",
        "final_answer": "Given a long text, and 5 events which take place in the long text, each indicated by number identifier [] that represents the shuffled order, (e.g. [0], [2], etc.). Reorder the events according to the original order of the events in the long text. The events should be listed in descending order using identifiers, and the first event in original order should be list first, and the output format should be [] > [], e.g., [0] > [2]. Only response the reorder results, do not say any word or explain.\n\nLong text:\n{content}\nEvents: {events}\n\nGive the reorder results, only give the ordered identifiers of the five events [] > [] > [] > [] > []: "
    }
    encoding = tiktoken.get_encoding("cl100k_base")
    full_text = sample["content"]
    summaries = sample["summaries"]
    input_text = []
    for i in range(len(summaries)):
        input_text.append("[{}] ```{}```".format(i, summaries[i]))
    input_data = prompt["final_answer"].format(
        content=full_text, events="\n".join(input_text)
    )
    input_text = input_data
    answer = sample["answer"]
    
    # prediction
    encoded_input = encoding.encode(input_text)
    if len(encoded_input) > max_input_length:
        encoded_input = encoded_input[:max_input_length//2] + encoded_input[-max_input_length//2:]
        input_text = encoding.decode(encoded_input)

    completion = client.chat.completions.create(
        model=fine_tuned_model,
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user', 'content': input_text},
        ],
        temperature=0.0,
        top_p=1,
        max_tokens=32,
        frequency_penalty = 0,
        presence_penalty = 0
    )
    pred = completion.choices[0].message.content
    numbers = re.findall(r"\d+", pred)
    numbers = [int(y) for y in numbers]
    processed_pred = numbers
    return {"pred": processed_pred, "answer": answer, "output": pred}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument('--dataset', type=str, default='reportsumsort', help='name of the dataset.')
    parser.add_argument('--api_key', type=str, default=os.getenv('GPT3_API_KEY'))
    parser.add_argument('--debug_size', type=int, default=63)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--offset', type=int, default=512)
    parser.add_argument('--n_epochs', type=int, default=3)
    parser.add_argument('--learning_rate_multiplier', type=float, default=0.5)
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--output_file', default="output/gpt3.5_assis_only_b2048_o1024_lrm0.5_epoch3_bamboo.json", type=str)
    parser.add_argument('--resume', action="store_true", help='whether to resume the fine-tuned models.')
    parser.add_argument('--resume_file', type=str, default='bamboo.pkl', help='file path to store the fine-tuned models.')
    parser.add_argument('--baseline', action="store_true")
    
    args = parser.parse_args()
    if args.resume and not args.resume_file:
        parser.error("--resume requires --resume_file to be provided")
    if args.do_eval and not args.output_file:
        parser.error("--do_eval requires --output_file to be provided")
    if args.baseline and not args.output_file:
        parser.error("--baseline requires --output_file to be provided")
        
    return args

if __name__ == '__main__':
    args = parse_args()
    
    ftgpt = FTGPT(api_key=args.api_key, model_name=args.model_name)
    if args.dataset == 'reportsumsort':
        with open('datasets/bamboo/reportsumsort_16k.jsonl', 'r') as f:
            raw_datasets = f.readlines()
    if args.debug_size:
        raw_datasets = raw_datasets[:args.debug_size]
        
    if args.baseline:
        if args.model_name == 'gpt-3.5-turbo-0125':
            max_input_length = 16000
        results = []
        logging.info("Begin evaluation.")
        for i, data in tqdm.tqdm(enumerate(raw_datasets), total=len(raw_datasets), desc="Eval"): 
            datapoint = eval(data)
            result = get_prediction(args.model_name, datapoint, max_input_length)
            results.append(datapoint)
            with open(args.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result)+ "\n")
    else:
        if args.model_name == 'gpt-3.5-turbo-0125':
            max_input_length = 16000
        ftdataset = FTDataset(block_size=args.block_size, offset=args.offset)
        if args.resume and os.path.exists(args.resume_file):
            ftgpt.resume(args.resume_file)
        for i, data in tqdm.tqdm(enumerate(raw_datasets), total=len(raw_datasets), desc="Train"):
            datapoint = eval(data)
            if args.resume and i < len(ftgpt):
                ft_model_name = ftgpt[i]
                logging.info(f"Get the resumed model {ft_model_name}")
            else:
                num_datapoints = dump_data(ftdataset, None, datapoint['content'])
                logging.info("Begin fine-tuning GPT.")
                ft_model_name = ftgpt.get_ft_model('output/chat_data.jsonl', hyperparameters={'batch_size': num_datapoints, 'n_epochs': args.n_epochs, 'learning_rate_multiplier': args.learning_rate_multiplier})
                logging.info("Finish fine-tuning GPT.")
                with open(args.resume_file, 'wb') as file:
                    logging.info(ftgpt.get_ft_models()[-1])
                    pickle.dump(ftgpt.get_ft_models(), file)

        if args.do_eval:
            results = []
            logging.info("Begin evaluation.")
            for i, data in tqdm.tqdm(enumerate(raw_datasets), total=len(raw_datasets), desc="Eval"):
                ft_model_name = ftgpt[i]
                datapoint = eval(data)
                result = get_prediction(ft_model_name, datapoint, max_input_length)
                results.append(datapoint)
                with open(args.output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result)+ "\n")
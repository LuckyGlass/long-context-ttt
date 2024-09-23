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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument('--dataset', type=str, default='datasets/loogle/shortdep_qa.jsonl', help='file path of the dataset.')
    parser.add_argument('--api_key', type=str, default=os.getenv('GPT3_API_KEY'))
    parser.add_argument('--debug_size', type=int)
    parser.add_argument('--block_size', type=int, default=2048)
    parser.add_argument('--offset', type=int, default=1024)
    parser.add_argument('--n_epochs', type=int, default=3)
    parser.add_argument('--learning_rate_multiplier', type=float, default=0.5)
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--output_file', default="output/debug.json", type=str)
    parser.add_argument('--resume', action="store_true", help='whether to resume the fine-tuned models.')
    parser.add_argument('--resume_file', type=str, default='short.pkl', help='file path to store the fine-tuned models.')
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
    ##todo:可能加一个外层循环，for longbench文件夹里的每个文件
    with open(args.dataset, 'r') as f:
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
            if 'qa_pairs' in datapoint and type(datapoint['qa_pairs']) == str:
                datapoint['qa_pairs'] = eval(datapoint['qa_pairs'])
            ftgpt.get_prediction(args.model_name, datapoint, max_input_length)
            results.append(datapoint)
            with open(args.output_file, 'w') as file:
                json.dump(results, file, indent=4)
    else:
        if args.model_name == 'gpt-3.5-turbo-0125':
            max_input_length = 16000
        ftdataset = FTDataset(block_size=args.block_size, offset=args.offset)
        if args.resume and os.path.exists(args.resume_file):
            ftgpt.resume(args.resume_file)
        for i, data in tqdm.tqdm(enumerate(raw_datasets), total=len(raw_datasets), desc="Train"):
            datapoint = eval(data)
            if 'qa_pairs' in datapoint and type(datapoint['qa_pairs']) == str:
                datapoint['qa_pairs'] = eval(datapoint['qa_pairs'])
            if args.resume and i < len(ftgpt):
                ft_model_name = ftgpt[i]
                logging.info(f"Get the resumed model {ft_model_name}")
            else:
                num_datapoints = dump_data(ftdataset, None, datapoint['context'])
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
                if 'qa_pairs' in datapoint and type(datapoint['qa_pairs']) == str:
                    datapoint['qa_pairs'] = eval(datapoint['qa_pairs'])
                ftgpt.get_prediction(ft_model_name, datapoint, max_input_length)
                results.append(datapoint)
                with open(args.output_file, 'w') as file:
                    json.dump(results, file, indent=4)
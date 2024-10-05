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
from gpt_jq import FTDataset, FTGPT, dump_data

# logging.basicConfig(
#     filename='logs/debug.log',
#     level=logging.INFO,
#     format='%(levelname)s - %(message)s',
#     filemode='w'
# )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument('--dataset', type=str, default='/scratch/nlp/lijiaqi/long-context-ttt1/datasets/longbench/', help='file path of the dataset.')
    parser.add_argument('--api_key', type=str, default="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")
    parser.add_argument('--debug_size', type=int, default=None)
    parser.add_argument('--block_size', type=int, default=2048)
    parser.add_argument('--offset', type=int, default=1024)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--learning_rate_multiplier', type=float, default=0.5)
    parser.add_argument('--do_eval', action="store_true", default=True)
    parser.add_argument('--training_file', default="./training/", type=str)
    parser.add_argument('--output_file', default="./output-gpt+ttt/", type=str)
    parser.add_argument('--resume', type=str, help='whether to resume the fine-tuned models.')
    parser.add_argument('--resume_file', type=str, default='./resume/', help='file path to store the fine-tuned models.')
    parser.add_argument('--baseline', type=str)
    
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


    dataset2prompt = json.load(open("/scratch/nlp/lijiaqi/long-context-ttt/scripts/longbench/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("/scratch/nlp/lijiaqi/long-context-ttt/scripts/longbench/dataset2maxlen.json", "r"))

    ftgpt = FTGPT(api_key=args.api_key, model_name=args.model_name)

    for root, dirs, files in os.walk(args.dataset):
        for dataset in files:

            prompt_ins = dataset2prompt[dataset.replace('.jsonl','')]
            max_gen = dataset2maxlen[dataset.replace('.jsonl','')]
            print('----------------------------------------------------------------------', dataset)

            with open(args.dataset+dataset, 'r') as f:
                raw_datasets = f.readlines() 
            if args.debug_size:
                raw_datasets = raw_datasets[:args.debug_size]
        
            if args.baseline == 'True':
                if args.model_name == 'gpt-3.5-turbo-0125':
                    max_input_length = 16000 - max_gen
                
                results = []
                logging.info("Begin evaluation.")
                for i, data in tqdm.tqdm(enumerate(raw_datasets), total=len(raw_datasets), desc="Eval"):
                    datapoint = json.loads(data)
                    
                    ftgpt.get_prediction(args.model_name, datapoint, prompt_ins, max_input_length, max_gen)
                    results.append(datapoint)
                    if not os.path.exists(args.output_file):
                        os.makedirs(args.output_file)
                    with open(args.output_file + dataset, 'w') as file:
                        json.dump(results, file, indent=4)
            else:
                if args.model_name == 'gpt-3.5-turbo-0125':
                    max_input_length = 16000 - max_gen

                ### ??? resume_file是个路径,文件可以包含多个pkl?

                if args.resume and os.path.exists(args.resume_file+dataset.split('.jsonl')[0]+'.pkl'):
                    ftgpt.resume(args.resume_file+dataset.split('.jsonl')[0]+'.pkl')


                # ftdataset = FTDataset(block_size=args.block_size, offset=args.offset)
                # for i, data in tqdm.tqdm(enumerate(raw_datasets), total=len(raw_datasets), desc="Train"):
                #     datapoint = json.loads(data)

                #     if args.resume and i < len(ftgpt):
                #         ft_model_name = ftgpt[i]
                #         logging.info(f"Get the resumed model {ft_model_name}")
                #     else:
                #         ### 一个json/任务,多个datapoint, 一个datapoint拆成多个training_data block都放在一个文件里?
                #         num_datapoints = dump_data(ftdataset, dataset, None, datapoint['context'], args.training_file )
                #         logging.info("Begin fine-tuning GPT.")
                        
                #         ###??????
                #         ft_model_name = ftgpt.get_ft_model(args.training_file + dataset, hyperparameters={'batch_size': num_datapoints, 'n_epochs': args.n_epochs, 'learning_rate_multiplier': args.learning_rate_multiplier})
                #         logging.info("Finish fine-tuning GPT.")
                #         if not os.path.exists(args.resume_file):
                #             os.makedirs(args.resume_file)
                #         with open(args.resume_file+dataset.split('.jsonl')[0]+'.pkl', 'wb') as file:
                #             logging.info(ftgpt.get_ft_models()[-1])
                #             pickle.dump(ftgpt.get_ft_models(), file)

                if args.do_eval:
                    results = []
                    logging.info("Begin evaluation.")
                    for i, data in tqdm.tqdm(enumerate(raw_datasets), total=len(raw_datasets), desc="Eval"):
                        ft_model_name = ftgpt[i]
                        datapoint = json.loads(data)
                        ftgpt.get_prediction(ft_model_name, datapoint, prompt_ins, max_input_length, max_gen)
                        results.append(datapoint)
                        if not os.path.exists(args.output_file):
                            os.makedirs(args.output_file)
                        with open(args.output_file + dataset, 'w') as file:
                            json.dump(results, file, indent=4)

    # ftgpt = FTGPT(api_key=args.api_key, model_name=args.model_name)
    # ftgpt.resume('./resume/musique.pkl')
    # print(len(ftgpt))
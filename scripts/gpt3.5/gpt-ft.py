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

logging.basicConfig(
    filename='logs/gpt3.5_assis_only_b2048_o1024_lrm0.5_epoch3_shortqa_eval.log',
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    filemode='w'
)

class FTDataset:
    def __init__(self, block_size: int, offset: int):
        self.data = []
        self.block_size = block_size
        self.offset = offset
        
    def process(self, datapoint):
        encoding = tiktoken.get_encoding("cl100k_base")
        text = "Title: " + datapoint['title'] + "\n" + "Text: " + datapoint['input']
        encoded_input = encoding.encode(text)
        length = len(encoded_input)
        training_data = [{"messages": [
            {"role": "assistant", "content": encoding.decode(encoded_input[s:min(s + self.block_size, length)])}
        ]} for s in range(0, length, self.offset) if len(encoded_input[s:]) > 0]
        return training_data
    

class FTGPT:
    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.ft_obj = []
    
    def resume(self, resume_path):
        with open(resume_path, 'rb') as f:
            self.ft_obj = pickle.load(f)
        logging.info(f"Resume ft_obj from {resume_path}; {len(self.ft_obj)} models in total.")

    def check_status(self):
        self.client.fine_tuning.jobs.list()
    
    def get_ft_model(self, training_file: str, **kwargs):
        file_obj = self.client.files.create(
            file=open(training_file, 'rb'),
            purpose='fine-tune',
        )
        ft_obj = self.client.fine_tuning.jobs.create(
            training_file=file_obj.id,
            model=self.model_name,
            **kwargs
        )
        
        job_id = ft_obj.id
        start_time = time.time()
        # check the status
        while True:
            try:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                if job.status == 'succeeded':
                    break
                elif job.status == 'failed':
                    raise Exception(f"Job {job_id} failed.")
                else:
                    logging.info(f"Job {job_id} is still in progress ({time.time() - start_time} seconds).")
                    time.sleep(60)
            except KeyboardInterrupt:
                self.client.fine_tuning.jobs.cancel(job_id)
                logging.warn(f"Cancel fine-tuning job {job_id}.")
                exit(0)
            except:
                print(job)
                self.client.fine_tuning.jobs.cancel(job_id)
                logging.error(f"Error in querying job {job_id}.")
                raise Exception(f"Error in querying job {job_id}.")
        logging.info(f"Job {job_id} succeeded in {time.time() - start_time} seconds.")
        
        try:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            for result_file in job.result_files:
                content = self.client.files.content(result_file).read().decode('utf-8')
                content = base64.b64decode(content).decode('utf-8')
                csv_reader = csv.reader(io.StringIO(content))
                for row in csv_reader:
                    logging.info(row)
        except:
            raise Exception(f"Error in retrieving result files from job {job_id}.")
        
        self.ft_obj.append(job)
        return job.fine_tuned_model
    
    def get_ft_models(self):
        return self.ft_obj
    
    def __getitem__(self, index):
        logging.info(f"Load model {index}: {self.ft_obj[index].fine_tuned_model}")
        return self.ft_obj[index].fine_tuned_model
    
    def __len__(self):
        return len(self.ft_obj)

    # for loogle
    def get_prediction(self, fine_tuned_model: str, datapoint, max_input_length: int, baseline: bool=False):
        
        max_gen = 500
        encoding = tiktoken.get_encoding("cl100k_base")
        for qa_pair in datapoint['qa_pairs']:
            if baseline:
                prompt_format = "Please answer the question based on the long texts from \"{title}\" below. \n{input}\nQuestion: {Q}\nAnswer: "
                prompt = prompt_format.format(title=datapoint["title"], input=datapoint['input'], Q=qa_pair['Q'])
                # clip the input to max_input_length
                encoded_input = encoding.encode(prompt)
                if len(encoded_input) > max_input_length:
                    encoded_input = encoded_input[:max_input_length//2] + encoded_input[-max_input_length//2:]
                    input_text = encoding.decode(encoded_input)
                else:
                    input_text = prompt

                completion = self.client.chat.completions.create(
                    model=fine_tuned_model,
                    messages = [
                        #{'role': 'system', 'content': "Please answer the question based on the long texts. If you are not sure about the answer, only output \'Not sure\'. "},
                        #{'role': 'user', 'content': input_text}
                        {'role': 'system', 'content': input_text},
                    ],
                    temperature=0.0,
                    top_p=1,
                    max_tokens=max_gen,
                    frequency_penalty = 0,
                    presence_penalty = 0
                )
                qa_pair['pred'] = completion.choices[0].message.content
            else:
                prompt_format = "Please answer the question based on the long texts from \"{title}\" below. \n{input}\nQuestion: {Q}\nAnswer: "
                prompt = prompt_format.format(title=datapoint["title"], input=datapoint['input'], Q=qa_pair['Q'])
                # clip the input to max_input_length
                encoded_input = encoding.encode(prompt)
                if len(encoded_input) > max_input_length:
                    encoded_input = encoded_input[:max_input_length//2] + encoded_input[-max_input_length//2:]
                    input_text = encoding.decode(encoded_input)
                else:
                    input_text = prompt

                completion = self.client.chat.completions.create(
                    model=fine_tuned_model,
                    messages = [
                        # {'role': 'system', 'content': "Please answer the question based on the long texts. If you are not sure about the answer, only output \'Not sure\'. "},
                        # {'role': 'user', 'content': input_text}
                        {'role': 'system', 'content': input_text},
                    ],
                    temperature=0.0,
                    top_p=1,
                    max_tokens=max_gen,
                    frequency_penalty = 0,
                    presence_penalty = 0
                )
                qa_pair['pred'] = completion.choices[0].message.content
                
                
# check the format of the dataset
def check_format(data_path: str):
    # Load the dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    # Initial dataset stats
    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)
    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")


def dump_data(datapoint, ftdataset, output_path='output/chat_data.jsonl'):
    training_data = ftdataset.process(datapoint)
    logging.debug(f"#Datapoints = {len(training_data)}")
    with open(output_path, 'w') as file:
        for data in training_data:
            file.write(json.dumps(data) + '\n')
    logging.info(f"Dump the datapoints into {output_path}")
    return len(training_data)

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
    parser.add_argument('--output_file', default="output/gpt3.5_assis_only_b2048_o1024_lrm0.5_epoch3_shortqa.json", type=str)
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
    with open(args.dataset, 'r') as f:
        raw_datasets = f.readlines() 
    if args.debug_size:
        raw_datasets = raw_datasets[:args.debug_size]
        
    if args.baseline:
        if args.model_name == 'gpt-3.5-turbo-0125':
            max_input_length = 16000
        with open(args.dataset, 'r') as f:
            raw_datasets = f.readlines() 
        if args.debug_size:
            raw_datasets = raw_datasets[:args.debug_size]
        results = []
        logging.info("Begin evaluation.")
        for i, data in tqdm.tqdm(enumerate(raw_datasets), total=len(raw_datasets), desc="Eval"):
            datapoint = eval(data)
            if type(datapoint['qa_pairs']) == str:
                datapoint['qa_pairs'] = eval(datapoint['qa_pairs'])
            ftgpt.get_prediction(args.model_name, datapoint, max_input_length, args.baseline)
            results.append(datapoint)
            with open(args.output_file, 'w') as file:
                json.dump(results, file, indent=4)
    else:
        if args.model_name == 'gpt-3.5-turbo-0125':
            max_input_length = 15000
        ftdataset = FTDataset(block_size=args.block_size, offset=args.offset)
        if args.resume and os.path.exists(args.resume_file):
            ftgpt.resume(args.resume_file)
        for i, data in tqdm.tqdm(enumerate(raw_datasets), total=len(raw_datasets), desc="Train"):
            datapoint = eval(data)
            if type(datapoint['qa_pairs']) == str:
                datapoint['qa_pairs'] = eval(datapoint['qa_pairs'])
            if args.resume and i < len(ftgpt):
                ft_model_name = ftgpt[i]
                logging.info(f"Get the resumed model {ft_model_name}")
            else:
                num_datapoints = dump_data(datapoint, ftdataset)
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
                if type(datapoint['qa_pairs']) == str:
                    datapoint['qa_pairs'] = eval(datapoint['qa_pairs'])
                ftgpt.get_prediction(ft_model_name, datapoint, max_input_length, args.baseline)
                results.append(datapoint)
                with open(args.output_file, 'w') as file:
                    json.dump(results, file, indent=4)
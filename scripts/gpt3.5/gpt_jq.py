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

class FTDataset:
    
    def __init__(self, block_size: int, offset: int):
        """
        block_size: 相当于论文初稿里定义的L
        offset: 相当于论文初稿里定义的S
        """
        self.data = []
        self.block_size = block_size
        self.offset = offset
        
    def process(self, title: str = None, input_text: str = None):
        """
        Args:
            title: 文章标题，可以为空，传入的话就在训练文章的开头加上标题
            input_text: 文章内容
        Returns:
            training_data: 一个list，每个元素是一个dict，dict的格式是
            {
                "messages": [
                    {"role": "assistant", "content": ...}
                ]
            }
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        if title:
            text = "Title: " + title + "\n" + "Text: " + input_text
        else:
            text = input_text
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
        """
        Args:
            resume_path (str): the path to the resume file
        """
        with open(resume_path, 'rb') as f:
            self.ft_obj = pickle.load(f)
        logging.info(f"Resume ft_obj from {resume_path}; {len(self.ft_obj)} models in total.")

    def check_status(self):
        self.client.fine_tuning.jobs.list()
    
    def get_ft_model(self, training_file: str, **kwargs):
        """
        Args:
            training_file (str): dump_data()生成的训练数据文件路径
        Returns:
            如果训练成功，返回可以直接使用的训练模型的编号，并存储在self.ft_obj列表中
        """
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
        "returns a list of the fine-tuned models"
        return self.ft_obj
    
    def __getitem__(self, index):
        """返回列表self.ft_obj中的第index个模型"""
        logging.info(f"Load model {index}: {self.ft_obj[index].fine_tuned_model}")
        return self.ft_obj[index].fine_tuned_model
    
    def __len__(self):
        """返回列表self.ft_obj的长度"""
        return len(self.ft_obj)

    # for loogle
    # 这个是针对loogl格式的，不适用于其他数据集
    def get_prediction(self, fine_tuned_model: str, datapoint, max_input_length: int):
        """
        Args:
            fine_tuned_model (str): 可以直接传递给openai的模型编号，既可以是微调过的模型也可是原始模型gpt3.5的名称
            datapoint (_type_): loogle里longqa和shortqa的一条数据
            max_input_length (int): 输入的最大长度，建议设置为16k
        Returns:
            因为datapoint是一个dict，作为引用传递，所以这个函数不返回任何值，而是直接修改datapoint里的内容
        """
        
        max_gen = 500
        encoding = tiktoken.get_encoding("cl100k_base")
        
        prompt_format = "Please answer the question based on the long texts from \"{title}\" below. \n{input}\nQuestion: {Q}\nAnswer: "
        prompt = prompt_format.format(title=None, input=datapoint['context'], Q=datapoint['input'])
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
        print(completion)
        datapoint['pred'] = completion.choices[0].message.content
                
                
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


def dump_data(ftdataset,data_name: str = None, title: str = None, input_text: str = None, output_path='output/'):
    """
    Args:
        ftdataset (_type_): 已经创建好的ftdataset类
        title (str, optional): 文章标题，传的话则训练数据加上标题
        input_text (str, optional): 文章内容
        output_path (str, optional): 生成的训练数据文件的路径，默认为'output/chat_data.jsonl'
    Returns:
        返回训练数据的长度，需要注意的是，训练数据总条数不能小于10条，不能大于256条，否则会报错。目前因为block size和offset较大，所以只单独处理了<10的情况
    """
    
    training_data = ftdataset.process(title, input_text)
    if len(training_data) < 10:
        blocksize = ftdataset.block_size
        offset = ftdataset.offset
        while len(training_data) < 10:
            blocksize = blocksize // 2
            offset = offset // 2
            new_ftdataset = FTDataset(block_size=blocksize, offset=offset)
            training_data = new_ftdataset.process(title, input_text)
    logging.debug(f"#Datapoints = {len(training_data)}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path+data_name, 'w') as file:
        for data in training_data:
            file.write(json.dumps(data) + '\n')
    logging.info(f"Dump the datapoints into {output_path}")
    return len(training_data)
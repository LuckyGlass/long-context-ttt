import json
from transformers import AutoTokenizer

with open("datasets/longbench/multifieldqa_en.jsonl", "r") as f:
    f.readline()
    data = json.loads(f.readline())

with open("datasets/longbench/multifieldqa_en_for_train.json", "w") as f:
    json.dump({
        'title': "Title",
        'input': data['context'],
        'qa_pairs': "None"
    }, f)

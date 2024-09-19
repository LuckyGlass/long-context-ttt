import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

with open('datasets/loogle/longdep_qa.jsonl') as f:
    raw_datasets = f.readlines()[20:]
sum = 0
for data in raw_datasets:
    data = eval(data)
    sum += len(encoding.encode(data["input"]))
print("Total training tokens for long qa: ", 4 * sum)

sum = 0
for data in raw_datasets:
    data = eval(data)
    data["qa_pairs"] = eval(data["qa_pairs"])
    sum += len(encoding.encode(data["input"])) * len(data["qa_pairs"])
    for qa_pair in data["qa_pairs"]:
        sum += len(encoding.encode(qa_pair["Q"]))
print("Total inferencing tokens for long qa: ", sum)

with open('datasets/loogle/shortdep_qa.jsonl') as f:
    raw_datasets = f.readlines()

sum = 0
for data in raw_datasets:
    data = eval(data)
    data["qa_pairs"] = eval(data["qa_pairs"])
    sum += len(encoding.encode(data["input"]))
print("Total training tokens for short qa: ", 4 * sum)

sum = 0
for data in raw_datasets:
    data = eval(data)
    data["qa_pairs"] = eval(data["qa_pairs"])
    sum += len(encoding.encode(data["input"])) * len(data["qa_pairs"])
    for qa_pair in data["qa_pairs"]:
        sum += len(encoding.encode(qa_pair["Q"]))
print("Total inferencing tokens for short qa: ", sum)
import json


with open("datasets/loogle/longdep_qa.jsonl", 'r') as f:
    data = list(map(json.loads, f.readlines()))

focus = {
    'José Luis Picardo': [0, 1, 2, 3, 4, 5],
    'Urban planning of Barcelona': [1, 2, 4, 6, 7],
    '2023 French pension reform unrest': [0, 1, 3, 4, 8, 9, 10],
    '2023 Turkey–Syria earthquake': [4, 5, 6, 7, 8, 9],
    'Claude List': [0, 1, 2, 3],
    'Climate change in Washington (state)': [0, 2, 3, 4]
}
# print([d['title'] for d in data])

for datapoint in data:
    if datapoint['title'] not in focus:
        continue
    print(datapoint['title'])
    ids = focus[datapoint['title']]
    qa_pairs = eval(datapoint['qa_pairs'])
    for i in ids:
        print(i, qa_pairs[i]['type'])

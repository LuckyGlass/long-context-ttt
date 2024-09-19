import os
import json
import tqdm

Q_types = ["comprehension_and_reasoning", "multiple_information_retrieval", "computation", "timeline_reorder"]
scores = {q_type: [] for q_type in Q_types}

with open("output/ttt_evidence_longqa-gpt.json", "r") as f:
    samples = json.load(f)
    
for sample in tqdm.tqdm(samples, total=len(samples)):
    for qa_pair in sample["qa_pairs"]:
        q_type = qa_pair["type"]
        scores[q_type].append(qa_pair["scores"]["gpt4_score"])

for q_type in Q_types:
    print(f"{q_type}: {sum(scores[q_type])/len(scores[q_type])}")
    
scores = {q_type: [] for q_type in Q_types}

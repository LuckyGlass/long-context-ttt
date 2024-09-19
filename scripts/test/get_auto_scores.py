import os 
import json
from nltk.translate.meteor_score import single_meteor_score
import tqdm
import argparse
import logging
from bert_score import BERTScorer

model_path = 'models/roberta-large'
bert_scorer = BERTScorer(model_type=model_path, device="cuda:0")

def get_bertscore(reference: str, pred: str):
    P, R, F1 = bert_scorer.score([pred], [reference], verbose=False)
    return F1.item()

def get_meteor_score(reference: str, pred: str):
    reference, pred = (
        reference.replace("\n", " ").split(),
        pred.replace("\n", " ").split(),
    )
    meteor = single_meteor_score(reference, pred)
    return meteor

def evaluate_qa(question: str, reference: str, pred: str):
    
    scores = {}
    scores['meteor'] = get_meteor_score(reference, pred)
    scores['bertscore'] = get_bertscore(reference, pred)

    return scores

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    
    args = parse_args()
    
    with open(args.input_file, 'r') as f:
        samples = json.load(f)
    
    results = {"meteor": [], "bertscore": []}
    new_samples = []
    for sample in tqdm.tqdm(samples, total=len(samples), desc='Scoring'):
        for qa_pair in sample['qa_pairs']:
            scores = evaluate_qa(qa_pair['Q'], qa_pair['A'], qa_pair['pred'])
            if "scores" in qa_pair.keys():
                qa_pair['scores']['meteor'] = scores['meteor']
                qa_pair['scores']['bertscore'] = scores['bertscore']
            if "gpt4_score" in qa_pair.keys():
                scores['gpt4_score'] = qa_pair['gpt4_score']
                qa_pair.pop('gpt4_score')
                qa_pair['scores'] = scores
            results['meteor'].append(scores['meteor'])
            results['bertscore'].append(scores['bertscore'])
        new_samples.append(sample)
    
    with open(args.output_file, 'w') as f:
        json.dump(new_samples, f, indent=4)
    file_name, file_ext = os.path.splitext(args.input_file)
    results['meteor'] = sum(results['meteor']) / len(results['meteor'])
    results['bertscore'] = sum(results['bertscore']) / len(results['bertscore'])
    with open(f"{file_name}-auto_score{file_ext}", 'w') as f:
        json.dump(results, f, indent=4)
            
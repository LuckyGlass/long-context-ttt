import os
import openai
import tqdm
import json
import argparse


client = openai.Client(api_key=os.getenv('GPT4_API_KEY'))

def get_gpt4_recall(reference: list, pred: str):
    gpt4_recall = []
    for evidence in reference:
        messages = [
            {'role': "system", 'content': "Given one target evidence, check whether it is in the recited evidences. Please only output 'Yes' or 'No'."},
            {'role': "user", 'content': "Target evidence: " + evidence +'\n' + "Recited evidences: " + pred + '\n'},
        ]
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=10,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        response = response.choices[0].message.content
        if 'Yes' in response or 'yes' in response or 'YES' in response:
            gpt4_recall.append(1)
        else:
            gpt4_recall.append(0)
    gpt4_recall = sum(gpt4_recall)/len(gpt4_recall)
    return gpt4_recall

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        data = json.load(f)
        
    gpt4_recalls = []
    new_data = []
    for datapoint in tqdm.tqdm(data, total=len(data), desc='Calculating GPT-4 recall'):
        for qa_pair in datapoint['qa_pairs']:
            if type(qa_pair["S"]) == str:
                evidences = [qa_pair["S"]]
            else:
                evidences = qa_pair["S"]
            gpt4_recall = get_gpt4_recall(evidences, qa_pair['S_recite'])
            qa_pair['gpt4_recall'] = gpt4_recall
            gpt4_recalls.append(gpt4_recall)
        new_data.append(datapoint)
        with open(args.output_file, 'w') as f:
            json.dump(data, f, indent=4)

    if len(gpt4_recalls) > 0:
        gpt4_recall = sum(gpt4_recalls)/len(gpt4_recalls)
        score_file = args.input_file.replace('.json', '-gpt4_recall.txt')
        with open(score_file, 'w') as f:
            f.write(f"gpt4_recall: {gpt4_recall}")

if __name__ == '__main__':
    main()
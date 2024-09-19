import os
import openai
import tqdm
import json
import argparse


client = openai.Client(api_key=os.getenv('GPT4_API_KEY'))

def get_gpt4_score(question: str, reference: str, pred: str):
    sys_prompt = "Given one question, there is a groundtruth and a predict_answer. Please decide whether they are the same or not in semantic. Please only output 'True' or 'False' ."
    prompt = [{"role": "system", "content": sys_prompt,},
    {
        "role": "user",
        "content": "Question: "
        + question
        + "\n"
        + "groundtruth = "
        + reference
        + "\n"
        + "predict_answer = "
        + pred,
    }]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=prompt,
        max_tokens=10,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    response = response.choices[0].message.content
    if not response:
        print("Error: No response from GPT-4")
    if 'True' in response or 'true' in response or 'TRUE' in response:
        return True
    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output JSON file.')
    args = parser.parse_args()
    
    file_name, file_ext = os.path.splitext(args.input)
    if file_ext == '.json':
        with open(args.input, 'r') as f:
            samples = json.load(f)
    elif file_ext == '.jsonl':
        samples = []
        with open(args.input, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
                
    gpt4_scores = []
    new_samples = []
    for sample in tqdm.tqdm(samples, total=len(samples), desc='Calculating GPT-4 score'):
        if "llm_output" in sample.keys():
            for i, qa_pair in enumerate(sample['question']):
                question = qa_pair["Q"]
                reference = qa_pair["A"]
                pred = sample['llm_output'][i]
                gpt4_score = get_gpt4_score(question, reference, pred)
                gpt4_scores.append(gpt4_score)
        else:
            for qa_pair in sample['qa_pairs']:
                question = qa_pair["Q"]
                reference = qa_pair["A"]
                pred = qa_pair["pred"]
                gpt4_score = get_gpt4_score(question, reference, pred)
                if 'scores' not in qa_pair.keys():
                    qa_pair['scores'] = {}
                    qa_pair['scores']['gpt4_score'] = gpt4_score
                else:
                    qa_pair['scores']['gpt4_score'] = gpt4_score
                gpt4_scores.append(gpt4_score)
            new_samples.append(sample)
            with open(args.output, 'w') as f:
                json.dump(new_samples, f, indent=4)
    if len(gpt4_scores) > 0:
        gpt4_score = sum(gpt4_scores) / len(gpt4_scores)
        file_name, file_extension = os.path.splitext(args.input)
        result_file = file_name + '-gpt4_score.txt'
        with open(result_file, 'w') as f:
            f.write(f"gpt4_score: {gpt4_score}")
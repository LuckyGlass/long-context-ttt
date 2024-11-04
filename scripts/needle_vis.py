"""
This code is adapted from https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/viz/CreateVizFromLLMTesting.ipynb.
"""
import os
import json
import tqdm
import openai
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from argparse import ArgumentParser


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


def load_pred_file(pred_file: str):
    with open(pred_file, 'r') as f:
        data = json.load(f)
    if 'length' not in data[0]:  # the first entry may be the config
        return data, 1
    else:
        return data, 0


def gpt_eval(data: list[dict], st_point: int, question: str, answer: str, cache_path: str):
    if os.path.exists(cache_path):
        print(f"Detect the existing file of GPT-score {cache_path}.")
        with open(cache_path, 'r') as f:
            results = json.load(f)
        return results
    for entry in tqdm.tqdm(data[st_point:], desc="GPT Eval"):
        entry['gpt_score'] = float(get_gpt4_score(question, answer, entry['pred']))
    print(f"Cache the file of GPT-score to {cache_path}.")
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=4)
    return data


def plot_data(eval_data: list[dict], st_point: int, output_path: str):
    table = [{'Document Depth': d['depth'], 'Context Length': d['length'], 'Score': d['gpt_score']} for d in eval_data[st_point:]]
    df = pd.DataFrame(table)
    print(df.head())
    print (f"You have {len(df)} rows")
    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    print(pivot_table.iloc[:5, :5])
    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        vmin=0.0,
        vmax=1.0
    )

    # More aesthetics
    plt.title('Pressure Testing GPT-4 128K Context\nFact Retrieval Across Context Lengths ("Needle In A HayStack")')  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Show the plot
    plt.savefig(output_path)
    # plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument('--pred_file', help="The path to the prediction file.")
    parser.add_argument('--output_path', help="The path to the plotted figure.")
    parser.add_argument('--question', help="The prompt of the test.")
    parser.add_argument('--answer', help="The needle, i.e., the ground truth.")
    args = parser.parse_args()
    pred_file_name, pred_file_ext = os.path.splitext(args.pred_file)
    cache_path = pred_file_name + '-gptscore' + pred_file_ext
    data, st_point = load_pred_file(args.pred_file)
    eval_data = gpt_eval(data, st_point, args.question, args.answer, cache_path)
    plot_data(eval_data, st_point, args.output_path)


if __name__ == '__main__':
    main()

import argparse
import jieba
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import pandas as pd
from rouge import Rouge
import seaborn as sns


class OpenAIEvaluator:
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""}

    def __init__(self,
                 model_name: str = "gpt-35-turbo-0125",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 true_answer: str = None,
                 question_asked: str = None,
                 proxy: str = None):
        """
        :param model_name: The name of the model.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """
        from langchain_community.chat_models import AzureChatOpenAI

        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked

        api_key = ""
        if (not api_key):
            raise ValueError("NIAH_EVALUATOR_API_KEY must be in env for using openai evaluator.")

        self.api_key = api_key
        REGION = ""
        API_BASE = ""
        
        self.evaluator = AzureChatOpenAI(model="gpt-35-turbo-0125",
                                         openai_api_key=self.api_key,
                                         openai_api_version="2024-02-01",
                                         azure_endpoint=f"{API_BASE}/{REGION}")

    def evaluate_response(self, response: str) -> int:
        from langchain.evaluation import load_evaluator

        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.evaluator,
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=self.true_answer,

            # The question asked
            input=self.question_asked,
        )

        return int(eval_result['score'])


def get_rouge_score(prediction, ground_truth):
    """
    https://github.com/THUDM/LongBench/blob/main/metrics.py
    """
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--gpt_eval', type=bool, default=True)
    parser.add_argument('--prompt', default="\n\nWhat is the best thing to do in San Francisco?\nAnswer:")
    parser.add_argument('--zh', type=bool, default=False)
    parser.add_argument('--test_length', nargs='*', default=[])
    parser.add_argument('--test_depth', nargs='*', default=[])
    args = parser.parse_args()

    with open(args.result_path, 'r') as f:
        results = json.load(f)
    
    if args.test_length is None:
        test_lengths = np.linspace(args.min_length, args.max_length, args.num_length_interval, endpoint=True).astype(int).tolist()
    else:
        test_lengths = args.test_length
    if args.test_depth is None:
        test_depths = np.linspace(args.min_depth, args.max_depth, args.num_depth_interval, endpoint=True).astype(int).tolist()
    else:
        test_depths = args.test_depth

    rouge_score = {l: {d: [] for d in v.keys()} for l, v in results.items()}
    if args.gpt_eval:
        evaluator = OpenAIEvaluator(question_asked=args.prompt.strip(), true_answer=args.needle.strip(), proxy=args.proxy)
        gpt_score = {l: {d: [] for d in v.keys()} for l, v in results.items()}

    for l, lv in results.items():
        for d, dv in lv.items():
            for v in dv:
                prediction = v["prediction"]
                target = v["target"]

                if args.zh:
                    score = get_rouge_score(' '.join(jieba.cut(prediction)), ' '.join(jieba.cut(target)))
                else:
                    score = get_rouge_score(prediction, target)
                rouge_score[l][d].append(score)

                if args.gpt_eval:
                    gpt_score[l][d].append(evaluator.evaluate_response(prediction))

            rouge_score[l][d] = round(sum(rouge_score[l][d]) / len(dv), 2)
            if args.gpt_eval:
                while 1:
                    try:
                        gpt_score[l][d] = round(sum(gpt_score[l][d]) / len(dv), 2)
                        break
                    except ValueError:
                        pass

    if args.gpt_eval:
        metrics = {'gpt': gpt_score}
    else:
        metrics = {'rouge': rouge_score}
    
    with open(os.path.join(args.output_dir, 'metric.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    for metric_key, metric_value in metrics.items():
        # Copied from https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/viz/CreateVizFromLLMTesting.ipynb
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
        # Create the heatmap with better aesthetics
        plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
        data = pd.DataFrame(metric_value)

        if metric_key == "rouge":
            vmin = 0
            vmax = 1
        elif metric_key == "gpt":
            vmin = 1
            vmax = 10

        ax = sns.heatmap(
            data,
            fmt="g",
            cmap=cmap,
            linewidths=0.1,
            linecolor='#808080',
            cbar_kws={'label': metric_key},
            vmin=vmin,
            vmax=vmax,
        )

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=24)
        cbar.set_label(label=f'{metric_key} score', fontdict={'size': 24})

        custom_x_labels = [str(item // 1024) + 'k' for item in test_lengths]
        custom_y_labels = [str(item) + '%' for item in test_depths]

        ax.set_xticklabels(custom_x_labels, rotation=45, ha='right', fontsize=24)
        ax.set_yticklabels(custom_y_labels, rotation=45, ha='right', fontsize=24)

        # More aesthetics
        # plt.title('Needle In A HayStack')  # Adds a title
        # plt.xlabel('Token Limit')  # X-axis label
        # plt.ylabel('Depth Percent')  # Y-axis label
        plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
        plt.yticks(rotation=45)  # Ensures the y-axis labels are horizontal
        plt.tight_layout()  # Fits everything neatly into the figure area
        # save to result_dir
        plt.savefig(os.path.join(args.output_dir, f"{metric_key}.pdf"), format='pdf', dpi=1200, bbox_inches='tight')


if __name__ == '__main__':
    main()

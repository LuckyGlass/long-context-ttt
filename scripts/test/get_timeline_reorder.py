import re
import json
import numpy as np
import argparse


def concordinary_index(pred_list, right_list):
    pred_set = set()
    right_set = set()
    error = False
    for i in range(len(pred_list) - 1):
        if min(pred_list) == 1:
            for j in range(i + 1, len(pred_list)):
                pred_set.add((pred_list[i] - 1, pred_list[j] - 1))
        else:
            for j in range(i + 1, len(pred_list)):
                pred_set.add((pred_list[i], pred_list[j]))
    if len(pred_list) != len(right_list):
        error = True
    for i in range(len(right_list) - 1):
        for j in range(i + 1, len(right_list)):
            right_set.add((right_list[i], right_list[j]))
    return len(right_set & pred_set) / len(right_set), error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred')
    args = parser.parse_args()
    with open(args.pred, 'r') as f:
        data = json.load(f)
    scores = []
    errors = []
    for sample in data:
        if 'qa_pairs' not in sample:
            continue
        for qa in sample['qa_pairs']:
            answer_pattern = r"\[[0-9]+\](?: < \[[0-9]+\])*"
            answer = list(map(lambda x: x + 1, qa['answers']))
            pred = re.findall(answer_pattern, qa['pred'])[0]
            pred = list(map(int, re.findall(r"[0-9]+", pred)))
            score, error = concordinary_index(pred, answer)
            if error:
                errors.append(1)
                scores.append(0)
            else:
                errors.append(0)
                scores.append(score)
            # print(answer, '->', pred, '=', score)
    print("Average score:", np.mean(scores))
    print("Error rate:", np.mean(errors))


if __name__ == '__main__':
    main()

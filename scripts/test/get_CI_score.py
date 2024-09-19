import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to the prediction file.')
    # parser.add_argument('--output', type=str, help='Path to the output file with CI score.')
    return parser.parse_args()

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

def evaluate(input_path, task):
    
    with open(input_path, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
        all_data = [json.loads(line) for line in lines]
        
    if task in ['reportsumsort','showssort']:
        count = 0
        all_count = 0
        format_error_count = 0
        for data in all_data:
            conr_index, error_flag = concordinary_index(data["pred"], data["answer"])
            if error_flag:
                format_error_count += 1
            else:
                count += conr_index
            all_count += 1
        print(
            "Cornordinary_index: {:.1f}".format(100 * count / all_count)
        )
        
    elif task in ['senhallu','abshallu']:
        # precision, recall, f1
        pred_list = []
        right_list = []
        for data in all_data:
            if "yes" in data["pred"].strip()[:3].lower():
                pred_list.append(True)
            else:
                pred_list.append(False)
            right_list.append(data['answer'])
        p, r, f = calculate_metrics(right_list, pred_list)
        print("precision/recall/f1: {:.1f}/{:.1f}/{:.1f}".format(100*p,100*r,100*f))

    elif task == "altqa":
        # accuracy
        all_count = 0
        count = 0
        for data in all_data:
            all_count += 1
            count += altqa_accuracy(data["pred"], data["answer"])
        print("accuracy: {:.1f}".format(100 * count / all_count))
        
    elif task in ['meetingpred','showspred']:
        # accuracy
        all_count = 0
        count = 0
        for data in all_data:
            all_count += 1
            count += pred_accuracy(data["pred"], data["answer"])
        print("accuracy: {:.1f}".format(100 * count / all_count))
        
    elif task in ['meetingqa','paperqa']:
        all_count = 0
        count = 0
        for data in all_data:
            all_count += 1
            count += qa_accuracy(data["pred"], data["answer"])
        print("accuracy: {:.1f}".format(100 * count / all_count))
        
    elif task == 'private_eval':
        evaluate_functional_correctness(input_path, problem_file='datasets/real_all_eval_v3.jsonl.gz')

def main():
    args = parse_args()
    evaluate(args.input, 'reportsumsort')
    
if __name__ == "__main__":
    main()
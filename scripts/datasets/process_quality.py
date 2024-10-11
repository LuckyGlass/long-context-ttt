"""
We use quality-train as the development dataset to tune the hyperparameters.
Useful properties: title, article, set_unique_id
Info: 300 articles in total, sample 50 articles. QA generation follows the same method as synthetic QA generation.
"""
import json
import numpy as np
import torch
import tqdm
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
# Configurations
data_path = 'datasets/QuALITY/QuALITY.v1.0.1.htmlstripped.train'
np.random.seed(0)
number_generated_qa = 2
number_article = 100


@torch.no_grad()
def shortqa_gen(generator, tokenizer, full_context: str, num_generate_qa: int=0):
    generated = []
    texts = sent_tokenize(full_context)
    c = 0
    assert len(texts) >= 25
    while len(generated) < num_generate_qa:
        st_pos = np.random.randint(0, len(texts) - 25)
        context = ' '.join(texts[st_pos:st_pos+25])
        messages = [
            {
                'role': "system",
                'content': "You are a helpful assistant."
            },
            {
                'role': "user", 
                'content': f"You are given a piece of text as the context. You should generate ONLY one question and the corresponding answer according to the context. You should also select one or more original sentences in the context as the evidences. Please answer in the following format:\nQuestion: [question]\nAnswer: [answer]\nEvidence:\n- [evidence 1]\n- [evidence 2]\n...\nPlease DON'T output quotes when outputing evidences. The following is the piece of text: {context}"
            }
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        mask_attention = torch.ones_like(input_ids)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        num_of_trials = 0
        while True:
            outputs = generator.generate(
                input_ids,
                max_new_tokens=1024,
                attention_mask=mask_attention,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=terminators,
                do_sample=False,
            )
            response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            question_position = response.find("Question:")
            answer_position = response.find("Answer:")
            evidence_position = response.find("Evidence:")
            question = response[question_position + 9:answer_position].strip()
            answer = response[answer_position + 7:evidence_position].strip()
            evidences = response[evidence_position + 9:].strip().split('\n')
            evidences = list(map(lambda s: s[s.find('-') + 2:].strip(), evidences))
            c += 1
            if question_position == -1 or answer_position == -1 or evidence_position == -1:
                num_of_trials += 1
                if num_of_trials > 5:
                    break
                continue
            else:
                question = response[question_position+9:answer_position].strip()
                answer = response[answer_position+7:evidence_position].strip()
                evidence = response[evidence_position+9:].strip()
                generated.append({"Q":question, "A":answer, "S": [evidence]})
                break
    return generated


def main():
    with open(data_path, 'r') as f:
        data = list(map(json.loads, f.readlines()))
    sampled_data = np.random.choice(data, size=number_article, replace=False)
    generator = AutoModelForCausalLM.from_pretrained('models/Meta-Llama-3-8B-Instruct')
    generator.eval()
    tokenizer = AutoTokenizer.from_pretrained('models/Meta-Llama-3-8B-Instruct')
    export_data = []
    for item in tqdm.tqdm(sampled_data):
        export_data.append({
            'set_unique_id': item['set_unique_id'],
            'title': item['title'],
            'input': item['article'],
            'qa_pairs': shortqa_gen(generator, tokenizer, item['article'], number_generated_qa)
        })
    with open('datasets/QuALITY/dev.json', 'w') as f:
        json.dump(export_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()

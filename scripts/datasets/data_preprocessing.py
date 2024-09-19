#!/usr/bin/env python
# coding=utf-8

import os
import json

with open("output/ttt-longqa-gpt2.json", "r") as f:
    samples = json.load(f)

new_samples = []
new_samples.append(samples[0])
for sample in samples[1:]:
    for qa_pair in sample["qa_pairs"]:
        qa_pair["pred"] = qa_pair["epochs=5.0; type=full"]["pred"]
        qa_pair["scores"] = qa_pair["epochs=5.0; type=full"]["scores"]
        qa_pair["perplexity"] = qa_pair["epochs=5.0; type=full"]["perplexity"]
        qa_pair.pop("epochs=5.0; type=full")
    new_samples.append(sample)

with open("output/ttt-longqa-gpt.json", "w") as f:
    json.dump(new_samples, f, indent=4)
    

with open("output/ttt-shortqa-gpt2.json", "r") as f:
    samples = json.load(f)

new_samples = []
new_samples.append(samples[0])
for sample in samples[1:]:
    for qa_pair in sample["qa_pairs"]:
        qa_pair["pred"] = qa_pair["epochs=5.0; type=full"]["pred"]
        qa_pair["scores"] = qa_pair["epochs=5.0; type=full"]["scores"]
        qa_pair["perplexity"] = qa_pair["epochs=5.0; type=full"]["perplexity"]
        qa_pair.pop("epochs=5.0; type=full")
    new_samples.append(sample)

with open("output/ttt-short-gpt.json", "w") as f:
    json.dump(new_samples, f, indent=4)
import csv
import json
import os

source_dir = './data/in_domain/test/source.txt'
target_dir = './runs/finetune/generated_predictions.txt'
output_dir = './runs/finetune/generated_predictions_ppl.json'
data = []

with open(source_dir) as f:
    lines = f.readlines()
    for line in lines:
        data.append({"text":line.strip('\n')})

with open(target_dir) as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        data[i]["text"] += line.strip('\n')

json_data = {"data": data}
json.dump(json_data, open(output_dir, 'w'),indent=2) 
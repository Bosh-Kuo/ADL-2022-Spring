import csv
import json
import os

dataset_dir = './data/in_domain/test'
output_dir = os.path.join(dataset_dir, 'text.json')

f = open(os.path.join(dataset_dir, 'source.txt'), 'w')
with open(os.path.join(dataset_dir, 'source.csv'), newline='') as csvfile:
    # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)
    data = []
    for row in rows:
        row.pop(0)
        text = "".join(row)
        f.write(text+'\n')
        data.append({"inputs":text})
f.close()

f = open(os.path.join(dataset_dir, 'target.txt'), 'w')
with open(os.path.join(dataset_dir, 'target.csv'), newline='') as csvfile:
    # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)
    for i, row in enumerate(rows):
        row.pop(0)
        text = "".join(row)
        f.write(text+'\n')
        data[i]["target"] = text
f.close()

json_data = {"data": data}
json.dump(json_data, open(output_dir, 'w'),indent=2)  
    
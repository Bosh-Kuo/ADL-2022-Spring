from random import shuffle
import sys
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm
from datasets import load_dataset

device = "cuda"
# gpt2, microsoft/DialoGPT-medium
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

prediction_ppl_dir = './runs/finetune/generated_predictions_ppl.json'
# prediction = load_dataset("text", data_files={"prediction": sys.argv[1]})["prediction"]
prediction = load_dataset('json', data_files={"prediction": prediction_ppl_dir}, field='data')["prediction"]  #'text.csv'

print(prediction)
loss = 0
steps = 0
nll = 0

for p in tqdm(prediction["text"]):
    if p:
        input_ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        #        shuffle(input_ids[0])
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nll += outputs[0].mean().item()
    steps += 1

average_nll = nll / steps
ppl = torch.exp(torch.tensor(average_nll)).item()

print(ppl)

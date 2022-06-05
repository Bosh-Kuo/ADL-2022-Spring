import json
import os
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import default_data_collator, Trainer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, AutoConfig
from transformers.trainer_callback import EarlyStoppingCallback
from tqdm import tqdm
from datasets import  load_dataset
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", default='./data/validation/hit_generator_text.json', type=str,
    )
    parser.add_argument(
        "--output_file", default='./data/validation/output.json', type=str, help="dir to save finetuned model"
    )
    parser.add_argument(
        "--model_name_or_path", default='./t5_hit_generator', type=str, help="model to finetune"  
    )
    parser.add_argument("--seed", default=26, type=int, help="random seed")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_kwargs = {
            "max_length": 32,
            "temperature": 0.65,
            "repetition_penalty":0.7,
            "num_return_sequences":3,
            "do_sample":True,
            "top_k":80,
            "top_p":0.95,
            # "num_beams": 5,
            # "do_sample": False,
        }

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    with open(args.data_file) as f:
        dataset = json.load(f)
    dataset = dataset["data"]

    for i in tqdm(range(len(dataset))):  #len(dataset)
        inputs = tokenizer(dataset[i]["source"], return_tensors="pt", padding=True).to(device)  


        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],         
            **gen_kwargs
        )
        output_ids = output_ids.cpu().numpy()
        text = [
            tokenizer.decode(candiate, skip_special_tokens=True) for candiate in output_ids
        ]
        dataset[i]["prediction"] = text[0]
        dataset[i]["candidates"] = text[1:]
        
        # print(dataset[i]["prediction"])
        # print(dataset[i]["candidates"])
        # print()
    
    json_data = {"data": dataset}
    json.dump(json_data, open(args.output_file, 'w'),indent=2)  





    


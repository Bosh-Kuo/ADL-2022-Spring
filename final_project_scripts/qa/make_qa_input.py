from argparse import ArgumentParser
from tqdm import tqdm
import jsonlines
import json

parser = ArgumentParser()
parser.add_argument("--data_file", type=str, default="../train_output.jsonl")
parser.add_argument("--output_file", type=str, default="qa_train_input.jsonl")
args = parser.parse_args()

samples = [json.loads(i) for i in open(args.data_file, "r")]

with jsonlines.open(args.output_file, mode='w') as writer:
    for dialog in tqdm(samples):
        output_dialog = {"dialog": [{"text":sentence} for sentence in dialog["dialog"]] }
        writer.write(output_dialog)
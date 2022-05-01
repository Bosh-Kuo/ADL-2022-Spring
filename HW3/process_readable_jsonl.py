import jsonlines
import os

if not os.path.exists('data_for_Read'):
    os.makedirs('data_for_Read')

sample_submission_dir = "./data/sample_submission.jsonl"
sample_submission_forRead_dir = "./data_for_Read/sample_submission_forRead.jsonl"
sample_test_dir = "./data/sample_test.jsonl"
sample_test_forRead_dir = "./data_for_Read/sample_test_forRead.jsonl"
public_dir = "./data/public.jsonl"
public_forRead_dir = "./data_for_Read/public_forRead.jsonl"
train_dir = "./data/train.jsonl"
train_forRead_dir = "./data_for_Read/train_forRead.jsonl"

dataPath = [(sample_submission_dir, sample_submission_forRead_dir), (sample_test_dir, sample_test_forRead_dir),
(public_dir, public_forRead_dir), (train_dir, (train_forRead_dir))]


for (rawDataPath, outputDataPath) in dataPath:
    with jsonlines.open(rawDataPath, mode='r') as reader:
        with jsonlines.open(outputDataPath, mode='w') as writer:
            for obj in reader:
                writer.write(obj)
            print(outputDataPath + " done!")


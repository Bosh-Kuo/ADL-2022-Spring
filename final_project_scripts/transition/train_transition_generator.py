import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import default_data_collator, Trainer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, AutoConfig
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_metric, load_dataset
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", default='./data', type=str
    )
    parser.add_argument(
        "--model_name_or_path", default='t5-small', type=str, help="model to finetune"  # t5-small
    )
    parser.add_argument(
        "--output_dir", default='t5_transition_generator', type=str, help="dir to save finetuned model"
    )
    parser.add_argument(
        "--max_epoch", default=100, type=int, help="total number of epoch"
    )
    parser.add_argument(
        "--train_bsize", default=16, type=int, help="training batch size"
    )
    parser.add_argument(
        "--eval_bsize", default=16, type=int, help="evaluation batch size"
    )
    args = parser.parse_args()
    return args

def preprocess_function(examples):
    inputs = [ex for ex in examples['source']]
    targets = [ex for ex in examples['target']]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True, padding='max_length',
        add_special_tokens=True,
    )

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, truncation=True, padding='max_length',
            add_special_tokens=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def read_data(data_dir):
    splits = ['train', 'validation', 'test']
    datasets = {}
    for split in splits:
        directory = os.path.join(data_dir, split)
        datasets[split] = load_dataset('json', data_files=f'{directory}/transition_text.json', field='data')  
        if split != 'test':
            datasets[split] = datasets[split].map(
                preprocess_function,
                batched=True,
                remove_columns=['source', 'target'],
            )['train']
        else:
            datasets[split] = datasets[split]['train']
    return datasets['train'], datasets['validation'], datasets['test']

if __name__ == "__main__":
    args = parse_args()
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path) 
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    max_input_length = 128
    max_target_length = 32

    # Dataset
    print('reading dataset')
    dataset_dir = args.dataset_root
    train_dataset, eval_dataset, test_dataset = read_data(dataset_dir) 
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    # Train model
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        num_train_epochs=args.max_epoch,
        per_device_train_batch_size=args.train_bsize,
        per_device_eval_batch_size=args.eval_bsize,
        label_smoothing_factor=0.1,
        eval_accumulation_steps=10,
        # weight_decay=0.01,               # strength of weight decay
    )
    
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # data_collator=default_data_collator,
        # compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.save_model(args.output_dir)
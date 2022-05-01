## Import packages
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)

from tw_rouge import get_rouge
from train_args import parse_args
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def main():
    ## Args
    args = parse_args()
    print("\n\n" + "="*100)
    print(args)

    ## Initialize the Accelerator and logger
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    print("\n\n" + "="*100)
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    ## Set seed and check output_dir
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    ## Load Dataset
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    raw_datasets = load_dataset("json", data_files=data_files)

    # Trim a number of training examples 取 100 個 data 來用，正式訓練或預測時關掉
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    print("\n\n" + "="*100)
    print(raw_datasets)


    ## Load pretrained model and tokenizer
    # config & tokenizer
    print("\n\n" + "="*100)
    if args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "Please make sure model_name_or_path is not None"
        )
        
    # mt5-small model
    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""


    ## Preprocessing the datasets.
    column_names = raw_datasets["train"].column_names
    dataset_columns = {"text_column": "maintext", "summary_column": "title"}
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    def preprocess_function(examples):
        if args.do_eval:
            inputs = examples[dataset_columns["text_column"]]
        elif args.do_train:
            inputs = examples[dataset_columns["text_column"]]
            targets = examples[dataset_columns["summary_column"]]
            
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs, max_length=args.max_source_length, padding="max_length", truncation=True)
        
        if args.do_train:
            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=args.max_target_length,
                                padding="max_length", truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss.
            if args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else label_pad_token_id) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def calculate_rouge(preds, refs, avg=True):
        preds = [pred.strip() +'\n' for pred in preds]
        refs = [ref.strip() +'\n' for ref in refs]
        return get_rouge(preds, refs, avg)

    # First we tokenize all the texts
    print("\n\n" + "="*100)
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    print("\n\n" + "="*100)
    print(train_dataset)
    print(eval_dataset)


    ## Create DataLoaders
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    ## Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps) 

    # args.max_train_steps = epoch * (num of data)/(batch size) / args.gradient_accumulation_steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )


    ## Prepare everything with our accelerator.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )


    ## Training
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_eval_rouge = 0
    eval_rouge_history = {"rouge-1":[], "rouge-2":[], "rouge-l":[]}

    for epoch in range(starting_epoch, args.num_train_epochs):
        # Train
        model.train()
        print("\n\n" + "="*100)
        print(f"..........Train - epoch{epoch}..........")
        for step, batch in enumerate(tqdm(train_dataloader)):

            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            # update parameters (weight and bias) 
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1
        
        # Evaluation
        model.eval()
        # greedy decoding by calling greedy_search() if num_beams=1 and do_sample=False.
        # multinomial sampling by calling sample() if num_beams=1 and do_sample=True.
        # beam-search decoding by calling beam_search() if num_beams>1 and do_sample=False.
        # beam-search multinomial sampling by calling beam_sample() if num_beams>1 and do_sample=True.
        # diverse beam-search decoding by calling group_beam_search(), if num_beams>1 and num_beam_groups>1.
        # constrained beam-search decoding by calling constrained_beam_search(), if constraints!=None or force_words_ids!=None.
        gen_kwargs = {
            "max_length": args.max_target_length,
            "num_beams": args.num_beams,
        }
        preds = []
        refs = raw_datasets["validation"]["title"]

        print("\n\n" + "="*100)
        print("..........Evaluate - epoch{epoch}..........")
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                generated_tokens = accelerator.gather(generated_tokens)
                generated_tokens = generated_tokens.cpu().numpy()
                
                # decode to sentences
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                preds += decoded_preds
        
        rouge_score = calculate_rouge(preds, refs, avg=True)
        eval_rouge_history["rouge-1"].append(rouge_score['rouge-1']['f'])
        eval_rouge_history["rouge-2"].append(rouge_score['rouge-2']['f'])
        eval_rouge_history["rouge-l"].append(rouge_score['rouge-l']['f'])
        eval_rouge = rouge_score['rouge-1']['f'] + rouge_score['rouge-2']['f'] + rouge_score['rouge-l']['f']
        
        print(f"rouge-1: {rouge_score['rouge-1']}")
        print(f"rouge-2: {rouge_score['rouge-2']}")
        print(f"rouge-l: {rouge_score['rouge-l']}")
        
        # store the best model
        if (eval_rouge > best_eval_rouge):
            best_eval_rouge = eval_rouge
            print("\n\n" + "="*100)
            if args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
    
    ## Save eval rouge history
    print("\n\n" + "="*100)
    print("..........Write eval_rouge_f1 file..........")
    with open(os.path.join(args.output_dir, "eval_rouge_f1.json"), "w") as f:
        json.dump(eval_rouge_history, f)

if __name__ == "__main__":
    main()




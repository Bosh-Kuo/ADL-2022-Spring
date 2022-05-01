## Import packages
import argparse
import json
import jsonlines
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
from predict_args import parse_args
from tw_rouge import get_rouge
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    ## Args
    args = parse_args()
    print("\n\n" + "="*100)
    print(args)


    ## Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
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
    accelerator.wait_for_everyone()


    ## Load pretrained model and tokenizer
    # config & tokenizer
    print("\n\n" + "="*100)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    prefix = args.source_prefix if args.source_prefix is not None else ""


    ## Load Dataset
    raw_datasets = load_dataset("json", data_files={"test": args.test_file})
    column_names = raw_datasets["test"].column_names
    dataset_columns = {"text_column": "maintext"}


    ## Preprocessing the datasets.
    def preprocess_function(examples):
        inputs = examples[dataset_columns["text_column"]]      
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs, max_length=args.max_source_length, padding="max_length", truncation=True)
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
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
    )
    test_dataset = processed_datasets["test"]
    print("\n\n" + "="*100)
    print(test_dataset)


    ## Create DataLoaders
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_test_batch_size
    )


    ## Prepare everything with our accelerator.
    model, test_dataloader = accelerator.prepare(model, test_dataloader)


    ## Test
    print("\n\n" + "="*100)
    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    
    # greedy decoding by calling greedy_search() if num_beams=1 and do_sample=False.
    # multinomial sampling by calling sample() if num_beams=1 and do_sample=True.
    # beam-search decoding by calling beam_search() if num_beams>1 and do_sample=False.
    # beam-search multinomial sampling by calling beam_sample() if num_beams>1 and do_sample=True.
    # constrained beam-search decoding by calling constrained_beam_search(), if constraints!=None or force_words_ids!=None.
    gen_kwargs = {
        "max_length": args.max_target_length,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
    }

    
    # Evaluation
    model.eval()
    preds = []
    for step, batch in enumerate(tqdm(test_dataloader)):
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
    

    ## write output jsonl file
    if not args.eval_rouge:
        print("\n\n" + "="*100)
        print("..........Write output file..........")
        with jsonlines.open(args.output_dir, mode='w') as writer:
            for Id, pred in zip(raw_datasets["test"]["id"], preds):
                writer.write({"title": pred, "id": Id})
        print("..........Finish !..........")


    ## evaluate rouge score
    if args.eval_rouge:
        refs = raw_datasets["test"]["title"]
        rouge_score = calculate_rouge(preds, refs, avg=True)
        print("\n\n" + "="*100)
        print(f"Strategy: {args.generation_method}")
        print("..........Write output file..........")
        with open(os.path.join("./strategy", f"{args.generation_method}.json"), "w") as f:
            json.dump(rouge_score, f)
        print("..........Finish !..........")
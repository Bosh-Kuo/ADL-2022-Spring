# ## Import packages
import argparse
import logging
import math
import os
import random
from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from utils_qa import postprocess_qa_predictions
from qa_args import parse_args



def main():
    args = parse_args()
    print("="*100 + "\n\n")
    print("Args:\n")
    print(args)

    # ## Initialize accelerator, logger
    logger = logging.getLogger(__name__)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()


    # ## Dataset

    # %%
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None and args.do_train:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None and args.do_eval:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if args.test_file is not None and args.do_predict:
            data_files["test"] = args.test_file
            extension = args.test_file.split(".")[-1]
        
        raw_datasets = load_dataset(extension, data_files=data_files, field="data")



    # ## Load pretrained model and tokenizer
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )

    # ## Preprocessing the datasets.

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    question_column_name = "question" 
    context_column_name = "context" 
    answer_column_name = "answers" 

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)


    # ## train preprocessing
    # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")  # ex: [0,1,2,2,3,4,4,4...]
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")  # ex: [off_0, off_1, off_2, off_2_1, off_3, off_4, off_4_1, off_4_2...]

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        #  若有些 QA context 太常會被分成好幾段，offset_mapping長度就是所有分段 QA input 的數量 
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            # None: 為特殊字符, 0: 該 token 屬於 question, 1: 該 token 屬於 context
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            # 第幾個 QA pair （一個 QA pair可以分成很多段，所以可能多段對應到同一個 QA pair）
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            # 此 QA 沒有給 answer就把 token start_positions, token end_positions 都定在cls的位置
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                # 找出該 span 第一個 context token 的位置
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                # 找出該 span 最後一個 context token 的位置
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                # 若答案不在一個過長的 QA 的其中一段，就把這段 token start_positions, token end_positions 都定在cls的位置
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # 確認 answer 一定在這段 span 的 context 中
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)  # 找出 answer start token 在該 span 的位置
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)  # 找出 answer end token 在該 span 的位置

        return tokenized_examples

    # ## create train dataset
    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        raw_train_dataset = raw_datasets["train"]

        # select train data
        if args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            raw_train_dataset = raw_train_dataset.select(range(args.max_train_samples))
        print("\n\n" + "="*100 + "\n\n")
        print(raw_train_dataset)

        # Create train feature from dataset
        with accelerator.main_process_first():
            train_dataset = raw_train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        print("\n\n" + "="*100 + "\n\n")
        print(train_dataset)  # 一個 QA pair 可能會被切好幾段，所以數量可能會大於 raw_train_dataset
            

    # ## Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")  # ex: [0,1,2,2,3,4,4,4...]

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            # 原本 question tokens 和 特殊字符 tokens 也有 offset pair (startIdx, endIdx), 將除了 context 之外的 offset mapping 都轉成 None 就能快速分辨哪些是 context
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    # ## create validation dataset
    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        raw_eval_dataset = raw_datasets["validation"]
        if args.max_eval_samples is not None:
            # We will select sample from whole data
            raw_eval_dataset = raw_eval_dataset.select(range(args.max_eval_samples))
        print("\n\n" + "="*100 + "\n\n")
        print(raw_eval_dataset)
        # Validation Feature Creation
        with accelerator.main_process_first():
            eval_dataset = raw_eval_dataset.map(
                prepare_validation_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=raw_datasets["validation"].column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        print("\n\n" + "="*100 + "\n\n")
        #  沒有 'start_positions', 'end_positions' 但有 'offset_mapping', 'example_id'
        print(eval_dataset)  # 一個 QA pair 可能會被切好幾段，所以數量可能會大於 raw_eval_dataset


    # ## create predict dataset
    if args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        raw_predict_dataset = raw_datasets["test"]
        if args.max_predict_samples is not None:
            # We will select sample from whole data
            raw_predict_dataset = raw_predict_dataset.select(range(args.max_predict_samples))
        print("\n\n" + "="*100 + "\n\n")
        print(raw_predict_dataset)
        # Predict Feature Creation
        with accelerator.main_process_first():
            predict_dataset = raw_predict_dataset.map(
                prepare_validation_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=raw_datasets["test"].column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        print("\n\n" + "="*100 + "\n\n")
        #  沒有 'start_positions', 'end_positions' 但有 'offset_mapping', 'example_id'
        print(predict_dataset)  # 一個 QA pair 可能會被切好幾段，所以數量可能會大於 raw_eval_dataset


    # ## DataLoaders creation
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    if args.do_train and args.do_eval:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )

        eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
        eval_dataloader = DataLoader(
            eval_dataset_for_model, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        metric = load_metric("squad_v2" if args.version_2_with_negative else "squad")

    if args.do_predict:
        predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
        predict_dataloader = DataLoader(
            predict_dataset_for_model, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )

    # ## Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=args.output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        if stage == "eval":
            references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)
        elif stage == "predict":
            return formatted_predictions



    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    # ## prepare hyperparameters
    # Optimizer
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

    if args.do_train and args.do_eval:
        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

    if args.do_predict:
        model, optimizer, predict_dataloader = accelerator.prepare(
        model, optimizer, predict_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    if args.do_train:    
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )


    # ## Train
    if args.do_train and args.do_eval:
        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        print("\n\n" + "="*100 + "\n\n")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        eval_EM = []
        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break


            # Evaluation
            print("\n\n" + "=" * 100 + "\n\n")
            logger.info("***** Running Evaluation *****")
            logger.info(f"  Num examples = {len(eval_dataset)}")
            logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

            all_start_logits = []
            all_end_logits = []
            for step, batch in enumerate(eval_dataloader):
                model.eval()
                with torch.no_grad():
                    outputs = model(**batch)
                    start_logits = outputs.start_logits
                    end_logits = outputs.end_logits

                    if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                        start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                        end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

                    all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
                    all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

            max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

            # concatenate the numpy array
            start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
            end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

            # delete the list of numpy arrays
            del all_start_logits
            del all_end_logits

            outputs_numpy = (start_logits_concat, end_logits_concat)
            prediction = post_processing_function(raw_eval_dataset, eval_dataset, outputs_numpy)
            eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
            logger.info(f"Evaluation metrics: {eval_metric}")
            eval_EM.append(eval_metric["exact_match"])

            # save check point
            if args.output_dir and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

        # save model
        if args.output_dir is not None:
            np.save(f'{args.output_dir}/eval_EM', np.array(eval_EM))
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)



    # ## Predict
    # Prediction
    def ans_postprocess(prediction_str):
        if '「' in prediction_str and '」' not in prediction_str:
            prediction_str += '」'
        elif '「' not in prediction_str and '」' in prediction_str:
            prediction_str += '」'
        return prediction_str.replace(',','')

    if args.do_predict:
        logger.info("***** Running Prediction *****")
        logger.info(f"  Num examples = {len(predict_dataset)}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

        all_start_logits = []
        all_end_logits = []
        print("..........Predict..........")
        for step, batch in enumerate(predict_dataloader):
            model.eval()
            with torch.no_grad():
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                    start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                    end_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)

                all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)

        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)

        # [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        predictions = post_processing_function(raw_predict_dataset, predict_dataset, outputs_numpy, stage="predict")

        # write csv file
        print("..........Write csv file..........")
        with open(args.output_csv, 'w') as f:
            f.write('id,answer\n')
            for i, prediction in enumerate(tqdm(predictions)):
                answer = ans_postprocess(prediction["prediction_text"])
                f.write(f'{prediction["id"]},{answer}\n')



if __name__ == "__main__":
    main()
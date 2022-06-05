import argparse
import json
import random

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transition_model_name_or_path",
        default="./transition/t5_transition_generator",
        type=str,
        help="model to chat with simulator",
    )
    parser.add_argument(
        "--hit_generator_model_name_or_path",
        default="./hit_generator/t5_hit_generator",
        type=str,
        help="model to mock user for generating hit utterance",
    )

    parser.add_argument("--num_chats", default=5, type=int, help="the number of round")
    parser.add_argument("--split", default="test", type=str, help="split")
    parser.add_argument("--seed", default=26, type=int, help="random seed")
    parser.add_argument(
        "--interactive_mode",
        action="store_true",
        help="make the simualtor interact with the user (type 'stop' to stop the program, type 'exit' to leave current round)",
    )
    parser.add_argument(
        "--output",
        default="output.jsonl",
        type=str,
        help="file to save the dialogs",
    )
    parser.add_argument(
        "--disable_output_dialog",
        action="store_true",
        help="whether output the dialogs to the command line",
    )
    args = parser.parse_args()
    return args


def preprocess(example):
    example["personas"] = [f"your persona: {p}" for p in example["personas"]]
    example["context"] = "\n".join(
        example["personas"]
        + (["additional_context"] if example["additional_context"] else [])
        + example["previous_utterance"]
    )

    return example


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mname = "facebook/blenderbot-400M-distill"
    simulator = BlenderbotForConditionalGeneration.from_pretrained(mname).to(device)
    simulator_tokenizer = BlenderbotTokenizer.from_pretrained(mname)

    # load your bot
    # transition bot
    bot = AutoModelForSeq2SeqLM.from_pretrained(args.transition_model_name_or_path).to(device)
    bot_tokenizer = AutoTokenizer.from_pretrained(args.transition_model_name_or_path)
    # hit ance generator
    hit_generator  = AutoModelForSeq2SeqLM.from_pretrained(args.hit_generator_model_name_or_path).to(device)
    hit_tokenizer = AutoTokenizer.from_pretrained(args.hit_generator_model_name_or_path)

    # generation kargs
    gen_kwargs = {
        "max_length": 32,
        "temperature": 0.65,
        "repetition_penalty":0.7,
        "do_sample":True,
        "top_k":80,
        "top_p":0.95,
        # "num_beams": 5,
        # "do_sample": False,
    }

    dataset = load_dataset("blended_skill_talk", split=args.split)
    dataset = dataset.map(
        preprocess,
        remove_columns=[
            "free_messages",
            "guided_messages",
            "suggestions",
            "personas",
            "additional_context",
            "previous_utterance",
        ],
    )

    if args.interactive_mode:
        for _ in range(args.num_chats):
            dialog = ["hi"]
            while True:
                inputs = simulator_tokenizer(
                    ["</s> <s>".join(dialog[-3:])], return_tensors="pt", truncation=True
                ).to(device)
                reply_ids = simulator.generate(**inputs, do_sample=True, top_p=0.8)
                text = simulator_tokenizer.batch_decode(
                    reply_ids, skip_special_tokens=True
                )[0].strip()
                dialog.append(text)
                print(f"\033[0;32;49m {'simulator: ': ^11}{text} \033[0;0m")

                text = input(f"\033[0;33;49m {'you: ': ^11}")
                dialog.append(text)
                if text in ["stop", "exit"]:
                    break
            if text == "stop":
                break
            print()
    else:
        assert args.num_chats <= len(
            dataset
        ), f"--num_chats needs to be smaller than dataset (<={len(dataset)})"
        dataset = dataset.select(random.sample(range(len(dataset)), args.num_chats))

        output = []
        for index, context in enumerate(
            tqdm(dataset["context"], disable=(not args.disable_output_dialog))
        ):
            dialog = []
            if not args.disable_output_dialog:
                print(f" dialog id: {index}")
            for _ in range(5):
                inputs = simulator_tokenizer(
                    [
                        "</s> <s>".join(
                            ([context] + dialog if len(dialog) < 3 else dialog[-3:])
                        )
                    ],
                    return_tensors="pt",
                    truncation=True,
                ).to(device)
                reply_ids = simulator.generate(**inputs)
                text = simulator_tokenizer.batch_decode(
                    reply_ids, skip_special_tokens=True
                )[0].strip()
                dialog.append(text)
                if not args.disable_output_dialog:
                    print(f"\033[0;32;49m {'simulator: ': ^11}{text} \033[0;0m")

                # you might need to change this line due to the model you use
                # predict hit_utterance
                inputs = hit_tokenizer(
                    " ".join(dialog[-3:]), return_tensors="pt", truncation=True, padding=True
                ).to(device)
                output_ids = hit_generator.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],         
                    **gen_kwargs
                )
                hit_utterance_text = hit_tokenizer.decode(output_ids[0], skip_special_tokens=True)

                # generate transition turn 
                inputs = bot_tokenizer(
                    f"{text} <blank> {hit_utterance_text}", return_tensors="pt", truncation=True, padding=True
                ).to(device)  
                output_ids = bot.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],         
                    **gen_kwargs
                )
                text = bot_tokenizer.decode(output_ids[0], skip_special_tokens=True)
                dialog.append(text)
                if not args.disable_output_dialog:
                    print(f"\033[0;33;49m {'bot: ': ^11}{text} \033[0;0m")

            output.append(dialog)
            if not args.disable_output_dialog:
                print()

        with open(args.output, "w") as f:
            for idx, dialog in enumerate(output):
                f.write(json.dumps({"id": idx, "dialog": dialog}) + "\n")

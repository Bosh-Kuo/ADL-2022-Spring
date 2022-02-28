import json
import logging
import pickle
import re
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from random import random, seed
from typing import List, Dict
import torch
from tqdm.auto import tqdm

from utils import Vocab

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--glove_path",
        type=Path,
        help="Path to Glove Embedding.",
        default="./glove.840B.300d.txt",
    )
    parser.add_argument("--rand_seed", type=int,
                        help="Random seed.", default=13)
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="Number of token in the vocabulary",
        default=10_000,
    )
    args = parser.parse_args()
    return args


def main(args):
    seed(args.rand_seed)

    intents = set()
    words = Counter()
    for split in ["train", "eval"]:
        dataset_path = args.data_dir / f"{split}.json"
        dataset = json.loads(dataset_path.read_text())
        logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")
        intents.update({instance["intent"] for instance in dataset})
        words.update(
            [token for instance in dataset for token in instance["text"].split()]
        )

    common_words = {w for w, _ in words.most_common(args.vocab_size)}
    # vocab = Vocab(common_words)
    # with open('test.pickle', 'wb') as f:
    #     pickle.dump(my_dict, f)
    
    glove: Dict[str, List[float]] = {}
    print(glove)


if __name__ == "__main__":
    args = parse_args()
    # parents: 父目錄不存在的話是否要創建父目錄, exist_ok: 只有在目錄不存在時創建目錄
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)

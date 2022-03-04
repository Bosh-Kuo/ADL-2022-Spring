import json
import logging
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from random import seed

from preprocess_intent import build_vocab  # 建立字典方式跟intent_classification一樣

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(args):
    seed(args.rand_seed)

    tags = set()
    words = Counter()
    for split in ["train", "eval"]:
        dataset_path = args.data_dir / f"{split}.json"
        dataset = json.loads(dataset_path.read_text())
        logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")

        # tags蒐集data中所有出現過的tags字串:{'B-time', 'I-people', ...}, length = 9 for both train/eval
        tags.update({tag for instance in dataset for tag in instance["tags"]})
        # words蒐集data中所有text裡的斷詞與對應的出現次數: Counter({'my': 5528, 'i': 5437,...})
        words.update(
            [token for instance in dataset for token in instance["tokens"]])

    # 將intents中的字串加上index(labelinig):{'B-time': 0, 'I-people': 1, ...}
    tag2idx = {tag: i for i, tag in enumerate(tags)}
    tag_idx_path = args.output_dir / "tag2idx.json"
    tag_idx_path.write_text(json.dumps(tag2idx, indent=2))
    logging.info(f"Tag 2 index saved at {str(tag_idx_path.resolve())}")

    build_vocab(words, args.vocab_size, args.output_dir, args.glove_path)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
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
        default="./cache/slot/",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="Number of token in the vocabulary",
        default=10_000,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)

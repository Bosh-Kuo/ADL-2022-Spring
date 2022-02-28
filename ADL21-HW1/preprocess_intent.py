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


def build_vocab(
    words: Counter, vocab_size: int, output_dir: Path, glove_path: Path
) -> None:
    # most_common會使用數量作排序，並且指定要排出前幾名，train/eval中token都不超過預設的10000
    common_words = {w for w, _ in words.most_common(vocab_size)}
    # 代表train/eval data蒐集來的tokens加上"[PAD]", "[UNK]"後的 tokens集
    vocab = Vocab(common_words)
    # pickle是專門支援Python的資料型態，若儲存資料為dict，就跟json一樣，若儲存資料為object，可以將其變數化儲存，下次使用時即可還原此object的工作狀態
    vocab_path = output_dir / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)   # 將vocab object存起來
    logging.info(f"Vocab saved at {str(vocab_path.resolve())}")

    glove: Dict[str, List[float]] = {}
    logging.info(f"Loading glove: {str(glove_path.resolve())}")
    with open(glove_path) as fp:
        row1 = fp.readline()
        # if the first row is not header
        if not re.match("^[0-9]+ [0-9]+$", row1):
            # seek to 0
            fp.seek(0)
        # otherwise ignore the header

        for i, line in tqdm(enumerate(fp)):
            cols = line.rstrip().split(" ")
            # glove文件每行第一個string為一個word
            word = cols[0]
            # 剩餘的string list為該word的vector(dim = 300)
            vector = [float(v) for v in cols[1:]]

            # skip word not in words if words are provided(只挑出train/eval data中出現過的word)
            if word not in common_words:
                continue
            glove[word] = vector
            glove_dim = len(vector)

    assert all(len(v) == glove_dim for v in glove.values())
    assert len(glove) <= vocab_size

    # 計算train/eval中有出現在glove裡面的token
    num_matched = sum([token in glove for token in vocab.tokens])
    # for intent: Token covered: 5435 / 6491 = 0.8373132028963179
    logging.info(
        f"Token covered: {num_matched} / {len(vocab.tokens)} = {num_matched / len(vocab.tokens)}"
    )
    # 將glove中與train/eval重疊的token的vector存成一個embeddings list，順序依照{"{PAD]": 0, "[UNK]": 1, "my": 2, "i": 3, ...}
    embeddings: List[List[float]] = [
        # dict.get(key[, value]) : value to be returned if the key is not found
        glove.get(token, [random() * 2 - 1 for _ in range(glove_dim)])
        for token in vocab.tokens
    ]
    embeddings = torch.tensor(embeddings)
    embedding_path = output_dir / "embeddings.pt"
    torch.save(embeddings, str(embedding_path))
    logging.info(f"Embedding shape: {embeddings.shape}")
    logging.info(f"Embedding saved at {str(embedding_path.resolve())}")


def main(args):
    seed(args.rand_seed)

    intents = set()
    words = Counter()
    for split in ["train", "eval"]:
        dataset_path = args.data_dir / f"{split}.json"
        dataset = json.loads(dataset_path.read_text())  # dataset為一個list
        # dataset_path.resolve()為train.json/eval/.json的絕對路徑
        logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")

        # intents蒐集data中所有出現過的intent字串:{'income', 'timezone', ...}, length = 150 for both train/eval
        intents.update({instance["intent"] for instance in dataset})
        # words蒐集data中所有text裡的斷詞與對應的出現次數: Counter({'my': 5528, 'i': 5437,...})
        words.update(
            [token for instance in dataset for token in instance["text"].split()]
        )

    # 將intents中的字串加上index:{'income': 0, 'timezone': 1, ...}
    intent2idx = {tag: i for i, tag in enumerate(intents)}
    intent_tag_path = args.output_dir / "intent2idx.json"
    intent_tag_path.write_text(json.dumps(intent2idx, indent=2))
    logging.info(f"Intent 2 index saved at {str(intent_tag_path.resolve())}")

    build_vocab(words, args.vocab_size, args.output_dir, args.glove_path)


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


if __name__ == "__main__":
    args = parse_args()
    # parents: 父目錄不存在的話是否要創建父目錄, exist_ok: 只有在目錄不存在時創建目錄
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)

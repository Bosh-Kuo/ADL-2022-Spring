import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def predict(args, model, dataloader, dataset):
    model.eval()
    predict = []

    for i, inputs in enumerate(dataloader):
        inputs = inputs.to(args.device)
        outputs = model(inputs)
        test_pred = [dataset.idx2label(idx.item())
                     for idx in torch.argmax(outputs, dim=-1)]
        for intent in test_pred:
            predict.append(intent)

        with open(args.pred_file, 'w') as f:
            f.write('id,intent\n')
            for i, y in enumerate(predict):
                f.write('test-{},{}\n'.format(i, y))


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(
        dataset, args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    # dev_data = json.loads(args.dev_file.read_text())
    # dev_dataset = SeqClsDataset(dev_data, vocab, intent2idx, args.max_len)
    # dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        args.packed_seq,
    )

    # load weights into model
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    model.to(args.device)
    model.eval()

    # TODO: predict dataset
    predict(args, model, dataloader, dataset)
    # TODO: write prediction to file (args.pred_file)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json",
        # required=True
    )
    # parser.add_argument(
    #     "--dev_file",
    #     type=Path,
    #     help="Path to the test file.",
    #     default="./data/intent/eval.json",
    #     # required=True
    # )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/best-model.pth",
        # required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    # PACK_PADDED_SEQUENCE
    parser.add_argument("--packed_seq", type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from dataset import SlotTagDataset
from model import SlotTagger
from utils import Vocab

def predict(args, model, dataloader, dataset):
    model.eval()
    label_perd = []

    for i, (inputs,tokens_length) in enumerate(dataloader):
        inputs = inputs.to(args.device)
        outputs = model(inputs)
        
        unpaded_label_pred = [torch.argmax(outputs , dim = -1)[i,:tokens_length[i]].tolist() for i in range(len(outputs))]
        for idx_sentence in unpaded_label_pred:
            label_perd.append([dataset.idx2label(idx) for idx in idx_sentence])
    
    print(label_perd)
    with open(args.pred_file, 'w') as f:
        f.write('id,tags\n')
        for i, sentence in enumerate(label_perd):
            f.write('test-{},'.format(i))
            for idx, tag in enumerate(sentence):
                if idx < len(sentence) - 1:
                    f.write('{} '.format(sentence[idx]))
                else:
                    f.write('{}\n'.format(sentence[idx]))
                    break


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    slot_idx_path = args.cache_dir / "tag2idx.json"
    slot2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SlotTagDataset(data, vocab, slot2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(
        dataset, args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    # dev_data = json.loads(args.dev_file.read_text())
    # dev_dataset = SlotTagDataset(dev_data, vocab, slot2idx, args.max_len)
    # dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SlotTagger(
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
    # predict(args, model, dev_dataloader, dev_dataset)
    # TODO: write prediction to file (args.pred_file)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json",
    )
    # parser.add_argument(
    #     "--dev_file",
    #     type=Path,
    #     help="Path to the test file.",
    #     default="./data/slot/eval.json",
    #     # required=True
    # )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/slot/best-model.pth",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)  # 128

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

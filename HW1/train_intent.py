import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from tqdm import trange
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args, model, train_dataloader, optimizer, criterion):
    model.train()
    totalData = 0
    correct_sum = 0;
    loss_sum = 0;
    for i, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        inputs = inputs.to(args.device)
        labels = labels.to(args.device, dtype=torch.long)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # gradient
        optimizer.step()  # update parameters
        
        batch_correct = torch.sum(torch.argmax(outputs , dim = -1) == labels).item() 
        correct_sum += batch_correct
        loss_sum +=loss.item()
        totalData += len(labels)
    print("\nTraining Dataset: ", totalData)
    return correct_sum/totalData, loss_sum/totalData
        

def validate(args, model, eval_dataloader, criterion, datasets):
    model.eval()
    totalData = 0
    correct_sum = 0;
    loss_sum = 0;
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(eval_dataloader):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device, dtype=torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_correct = torch.sum(torch.argmax(outputs , dim = -1) == labels).item() 
            correct_sum += batch_correct
            loss_sum += loss.item()
            totalData += len(labels)

            # ans = [datasets['eval'].idx2label(idx.item()) for idx in labels]
            # pred = [datasets['eval'].idx2label(idx.item()) for idx in torch.argmax(outputs , dim = -1)]

    print("\nDev Dataset: ", totalData)
    return correct_sum/totalData, loss_sum/totalData



def main(args):
    # Check GPU
    GPU_name = torch.cuda.get_device_name()
    print("My GPU is {}\n".format(GPU_name))

    # set seed
    same_seeds(args.seed)

    # load vocab object (preprocess後的工作狀態)
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    # load label dict
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())  # label_mapping

    # {'train': path of train data, 'eval': path of eval data}
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    # data['train']:[{'text': ..., 'intent': ..., 'id': ... }, {...}, ...]
    data = {split: json.loads(path.read_text())
            for split, path in data_paths.items()}
    
    # {'train': SeqClsDataset object, 'eval': SeqClsDataset object}
    # datasets['train'][0]: {'text': 'i need you to book me a flight from ft lauderdale to houston on southwest', 'intent': 'book_flight', 'id': 'train-0'}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets
    # {'train': DataLoader object, 'eval': DataLoader object}
    # inputs: [batch_size, max_len], labels: [batch_size]
    # input: 經過encode, padding的tokens list batch, label: intent2idx的idx batch
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) for split, split_dataset in datasets.items()
    }

    # 一個mapping matrix, 將輸入的token idx tensor轉成低維空間中的wordvector
    embeddings = torch.load(args.cache_dir / "embeddings.pt")


    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional,
                          datasets[TRAIN].num_classes, args.packed_seq).to(args.device)
    # model.train()
    # train_dataloader = dataloaders['eval']
    # print(len(train_dataloader))
    # inputs,  labels= next(iter(train_dataloader))
    # inputs = inputs.to(device)
    # outputs = model(inputs)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # TODO: Training loop - iterate over train dataloader and update model weights
    # TODO: Evaluation loop - calculate accuracy and save model weights
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_eval_acc = 0
    for epoch in epoch_pbar:    
        avg_acc_train, avg_loss_train = train(args, model, dataloaders[TRAIN], optimizer, criterion)
        print("Epoch: {} || Train Acc: {:.3f}, Train Loss: {:.3f}".format(epoch, avg_acc_train, avg_loss_train))
        avg_acc_dev, avg_loss_dev = validate(args, model, dataloaders[DEV], criterion, datasets)
        print("Epoch: {} || Dev Acc: {:.3f}, Dev Loss: {:.3f}".format(epoch, avg_acc_dev, avg_loss_dev))
        if(avg_acc_dev > best_eval_acc):
            best_eval_acc = avg_acc_dev
            best_ckpt_path = args.ckpt_dir / 'best-model.pth'
            torch.save(model.state_dict(), best_ckpt_path)
            print("Save best model to {} epoch".format(best_ckpt_path))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)  # 128
    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--seed", type=int, default=43)

    # PACK_PADDED_SEQUENCE
    parser.add_argument("--packed_seq", type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

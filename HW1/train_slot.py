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
from dataset import SlotTagDataset
from model import SlotTagger
from utils import Vocab
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

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
    for i, (inputs, labels, tokens_length) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        inputs = inputs.to(args.device)
        labels = labels.to(args.device, dtype=torch.long)
        outputs = model(inputs)
        
        # 累加預測全對的sentenc
        unpaded_label_true = [labels[i,:tokens_length[i]].tolist() for i in range(len(outputs))]
        unpaded_label_pred = [torch.argmax(outputs , dim = -1)[i,:tokens_length[i]].tolist() for i in range(len(outputs))]
        correct_sum += sum([unpaded_label_true[i] == unpaded_label_pred[i] for i in range(len(unpaded_label_pred))])
        totalData += len(labels)

        outputs = outputs.view(-1, outputs.shape[-1])  # [batch_size * seq_len, num of label] 
        labels = labels.view(-1)  # batch_size * seq_len
        loss = criterion(outputs, labels)
        loss.backward()  # gradient
        optimizer.step()  # update parameters
        loss_sum +=loss.item()
        
    print("\nTraining Dataset: ", totalData)
    return correct_sum/totalData, loss_sum/totalData

def validate(args, model, eval_dataloader, criterion, datasets):
    model.eval()
    totalData = 0
    correct_sum = 0;
    loss_sum = 0;
    label_true = []
    label_perd = []
    
    with torch.no_grad():
        for i, (inputs, labels, tokens_length) in enumerate(tqdm(eval_dataloader)):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device, dtype=torch.long)
            outputs = model(inputs)
            # loss = criterion(outputs, labels)

            # 累加預測全對的sentence
            unpaded_label_true = [labels[i,:tokens_length[i]].tolist() for i in range(len(outputs))]
            unpaded_label_pred = [torch.argmax(outputs , dim = -1)[i,:tokens_length[i]].tolist() for i in range(len(outputs))]
            correct_sum += sum([unpaded_label_true[i] == unpaded_label_pred[i] for i in range(len(unpaded_label_pred))])
            totalData += len(labels)
            
            for idx_sentence in unpaded_label_true:
                label_true.append([datasets[DEV].idx2label(idx) for idx in idx_sentence])
            for idx_sentence in unpaded_label_pred:
                label_perd.append([datasets[DEV].idx2label(idx) for idx in idx_sentence])
            

            outputs = outputs.view(-1, outputs.shape[-1])  # [batch_size * seq_len, num of label] 
            labels = labels.view(-1)  # batch_size * seq_len
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            
            
        print("\nDev Dataset: ", totalData)
        print(label_true[0:10])
        print(label_perd[0:10])
        # print(classification_report(label_true, label_perd, mode='strict', scheme=IOB2))
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
    slot_idx_path = args.cache_dir / "tag2idx.json"
    slot2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())  # label_mapping

    # {'train': path of train data, 'eval': path of eval data}
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    # data['train']:[{'tokens': ..., 'tags': ..., 'id': ... }, {...}, ...] 
    data = {split: json.loads(path.read_text())
            for split, path in data_paths.items()}

    # {'train': SeqClsDataset object, 'eval': SeqClsDataset object}
    # datasets['train'][0]: {'tokens': ['i', 'have', 'three', 'people', 'for', 'august', 'seventh'], 'tags': ['O', 'O', 'B-people', 'I-people', 'O', 'B-date', 'O'], 'id': 'train-0'}
    datasets: Dict[str, SlotTagDataset] = {
        split: SlotTagDataset(split_data, vocab, slot2idx, args.max_len)
        for split, split_data in data.items()
    }

    # {'train': DataLoader object, 'eval': DataLoader object}
    # inputs: [batch_size, max_len], labels: [batch_size, max_len], tokens_length: [batch_size]
    # input: 經過encode, padding的tokens list batch, label: 每個詞對應slot2idx的idx batch, tokens_length:每個句子padding前的長度
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(split_dataset, args.batch_size, shuffle=False, collate_fn=split_dataset.collate_fn) for split, split_dataset in datasets.items()
    }

    # 一個mapping matrix, 將輸入的token idx tensor轉成低維空間中的wordvector
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SlotTagger(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional,
                          datasets[TRAIN].num_classes, args.packed_seq).to(args.device)
    
    # train_dataloader = dataloaders['train']
    # inputs,  labels, tokens_length = next(iter(train_dataloader))
    # inputs = inputs.to(args.device)
    # labels = labels.to(args.device)
    # outputs = model(inputs)
    # print(inputs)
    # print(labels)
    # print(tokens_length)
    # print(outputs.size())
    # unpaded_label_idx_batch = [labels[i,:tokens_length[i]].tolist() for i in range(len(outputs))]
    # unpaded_label_batch = [[datasets[TRAIN].idx2label(idx) for idx in idx_sentence] for idx_sentence in unpaded_label_idx_batch]
    # print(unpaded_label_idx_batch)
    # print(unpaded_label_batch)
    # correct_sum = sum([unpaded_label_idx_batch[i] == unpaded_label_idx_batch[i] for i in range(len(unpaded_label_idx_batch))])
    # print(correct_sum)
    
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
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
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
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)

    # PACK_PADDED_SEQUENCE
    parser.add_argument("--packed_seq", type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
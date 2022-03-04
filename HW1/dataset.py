from typing import List, Dict
from torch.utils.data import Dataset
from utils import Vocab, pad_to_len
import torch

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],  # split_data: [{'test': ,'intent':, 'id': }, {'test': ,'intent':, 'id': }, ...]
        vocab: Vocab,
        label_mapping: Dict[str, int],  # 同intent2idx.json
        max_len: int,  # 句子最長長度
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # Tokenization
        batch_tokens = [s['text'].split() for s in samples]  # List[List[str]]
        # encoding and padding
        # 將batch_tokens encode成idx後做padding，長度為to_len
        batch_encoded_tokens = self.vocab.encode_batch(batch_tokens = batch_tokens , to_len = self.max_len)
        batch_encoded_tokens = torch.tensor(batch_encoded_tokens)
        if 'intent' in samples[0].keys():  # for train/eval dataset
            batch_label = [self.label2idx(s['intent']) for s in samples]
            batch_label = torch.tensor(batch_label)
            return batch_encoded_tokens, batch_label
        else:  # for test dataset
            return batch_encoded_tokens
            
    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SlotTagDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],  # split_data: [{'tokens': ,'tags':, 'id': }, {'tokens': ,'tags':, 'id': }, ...]
        vocab: Vocab,
        label_mapping: Dict[str, int],  # 同tag2idx.json
        max_len: int,  # 句子最長長度
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance
    
    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # Tokenization
        batch_tokens = [s['tokens'] for s in samples]  # List[List[str]]
        # encoding and padding
        # 將batch_tokens encode成idx後做padding，長度為to_len
        batch_encoded_tokens = self.vocab.encode_batch(batch_tokens = batch_tokens , to_len = self.max_len)
        tokens_length = [sum([token != self.vocab.pad_id for token in sentence]) for sentence in batch_encoded_tokens]  # .lt means less than
        batch_encoded_tokens = torch.tensor(batch_encoded_tokens)
        
        if 'tags' in samples[0].keys():  # for train/eval dataset
            batch_label = [[self.label2idx(tag) for tag in s['tags']] for s in samples]
            padded_batch_label = pad_to_len(batch_label, to_len=self.max_len, padding=self.label_mapping["PAD"])
            padded_batch_label = torch.tensor(padded_batch_label)
            return batch_encoded_tokens, padded_batch_label, tokens_length
        else:  # for test dataset
            return batch_encoded_tokens, tokens_length
    
    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
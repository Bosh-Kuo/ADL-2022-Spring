from typing import Iterable, List


class Vocab:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)},  # 從下標2開始
            # {"{PAD]": 0, "[UNK]": 1, "my": 2, "i": 3, ...}
        }

    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]  # 回傳"[PAD]"的idx:0

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]  # 回傳"[UNK]"的idx:1

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())  # 只回傳tokens

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)  # 回傳該token的idx，不存在的話就回傳[UNK]的idx

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]  # 回傳tokens list(一個句子) 的 idx list

    def encode_batch(
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        batch_ids = [self.encode(tokens) for tokens in batch_tokens]  # 一個batch的tokens list(句子)轉為一個batch的idx list: exp:[[1,3,4],[2,5,6,8]...]
        to_len = max(len(ids)
                     for ids in batch_ids) if to_len is None else to_len  # 沒有指定to_len的話就以batch中長度最長的句子的長度作為to_len
        padded_ids = pad_to_len(batch_ids, to_len, self.pad_id)   #  將每個句子都以"[PAD]"的idx padding到長度都為to_len
        return padded_ids


def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = [seq[:to_len] + [padding] *
               max(0, to_len - len(seq)) for seq in seqs]  #  將batch中的batch_ids以"[PAD]"的idx padding到每個句子長度都為to_len
    return paddeds  

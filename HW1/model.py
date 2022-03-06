from typing import Dict
import torch
from torch import nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        packed: bool
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embed_dim = embeddings.size(1)  # 輸入 x 的特徵數量，embedding dim = 300
        self.hidden_size = hidden_size  # 隱藏層 h 的特徵數量，
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.packed = packed
        self.rnn = nn.LSTM(self.embed_dim, self.hidden_size, self.num_layers,
                           dropout=dropout, bidirectional=self.bidirectional, batch_first=True)  # format:(batch, seq len, feature)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder_output_size, self.num_class)
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        # raise NotImplementedError
        if self.bidirectional:
            return 2 * self.hidden_size
        else:
            return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        if (self.packed):
            #每個句子padding前的長度[batch_size]
            input_length = torch.tensor([sum(input.gt(0)) for input in batch])  # .gt means grater
            # 將每個句子按未padding前的長度降冪排列 [batch_size]
            sorted_input_length, sorted_idx = input_length.sort(dim=0, descending=True)  
            sorted_tokens_idx = batch[sorted_idx]  # [batch_size, max_len]
            _, original_idx = sorted_idx.sort(dim=0)  # 原本的idx排序後在什麼位置, [batch_size]

            # 轉為word vector [batch_size, max_len, embed_dim]
            x = self.embed(sorted_tokens_idx) 
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths = sorted_input_length, batch_first=True)
            
            # packed_seq_out: [batch_size, seq_len, hiidden_size]
            # h_n, c_n: [bidirectional*numnum_layers, batch_size, hidden_size]
            packed_seq_out, (h_n, c_n) = self.rnn(packed_x)  
            seq_out, _ = nn.utils.rnn.pad_packed_sequence(packed_seq_out, batch_first=True, total_length = batch.size(1))
            # 合併雙向h_n [batch_size, 2*hidden_size]
            h_n = torch.cat((h_n[-1], h_n[-2]), axis=-1) if self.bidirectional else h_n[-1]
            output = self.classifier(h_n)
            output = torch.index_select(output, 0, original_idx.to('cuda'))  # 依照給定的idx序列在dim=0上取tensor
            return output
            # print('input_length: ', input_length)
            # print('sorted_input_length: ', sorted_input_length)
            # print('original input_length: ', torch.index_select(sorted_input_length,  0, original_idx))
            # print('sorted_idx ', sorted_idx)
            # print('original_idx: ', original_idx)
            # print(output.size())
        
        else:
            # 轉為word vector [batch_size, max_len, embed_dim]
            x = self.embed(batch)
            seq_out, (h_n, c_n) = self.rnn(x)
            h_n = torch.cat((h_n[-1], h_n[-2]), axis=-1) if self.bidirectional else h_n[-1]  # 選最後一層LSTM layer雙向輸出的h_n
            output = self.classifier(h_n)
            return output
            # print("output size: ",output.size())
            # print("h_n size: ",h_n.size())


class SlotTagger(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        packed: bool
    ) -> None:
        super(SlotTagger, self).__init__()
        self.embed = nn.Sequential(
            Embedding.from_pretrained(embeddings, freeze=False),              
            nn.Dropout(),
        )
        
        # TODO: model architecture
        self.embed_dim = embeddings.size(1)  # 輸入 x 的特徵數量，embedding dim = 300
        self.hidden_size = hidden_size  # 隱藏層 h 的特徵數量，
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.packed = packed
        self.rnn = nn.LSTM(self.embed_dim, self.hidden_size, self.num_layers,
                           dropout=dropout, bidirectional=self.bidirectional, batch_first=True)  # format:(batch, seq len, feature)
        self.batchNorm1d = nn.BatchNorm1d(self.encoder_output_size)
        self.classifier = nn.Sequential(            
            nn.Linear(self.encoder_output_size, self.encoder_output_size),
            nn.ReLU(),
            nn.Linear(self.encoder_output_size, self.num_class),
        )


    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        # raise NotImplementedError
        if self.bidirectional:
            return 2 * self.hidden_size
        else:
            return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        if (self.packed):
            #每個句子padding前的長度[batch_size]
            # print(batch)
            input_length = torch.tensor([sum(input.lt(9)) for input in batch])  # .lt means less than
            # print(input_length)
            # 將每個句子按未padding前的長度降冪排列 [batch_size]
            sorted_input_length, sorted_idx = input_length.sort(dim=0, descending=True)  
            sorted_tokens_idx = batch[sorted_idx]  # [batch_size, max_len]
            _, original_idx = sorted_idx.sort(dim=0)  # 原本的idx排序後在什麼位置, [batch_size]

            # 轉為word vector [batch_size, max_len, embed_dim]
            x = self.embed(sorted_tokens_idx) 
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths = sorted_input_length, batch_first=True)
            
            # packed_seq_out: [batch_size, seq_len, hiidden_size*bidirectional]
            # h_n, c_n: [bidirectional*numnum_layers, batch_size, hidden_size]
            packed_seq_out, (h_n, c_n) = self.rnn(packed_x)  
            seq_out, _ = nn.utils.rnn.pad_packed_sequence(packed_seq_out, batch_first=True, total_length = batch.size(1))
            # 合併雙向h_n [batch_size, 2*hidden_size]

            # batch norm
            seq_out = seq_out.permute(0,2,1)
            seq_out = self.batchNorm1d(seq_out)
            seq_out = seq_out.permute(0,2,1)

            output = self.classifier(seq_out)
            output = torch.index_select(output, 0, original_idx.to('cuda'))  # 依照給定的idx序列在dim=0上取tensor
            
            # print('input_length: ', input_length)
            # print('sorted_input_length: ', sorted_input_length)
            # print('original input_length: ', torch.index_select(sorted_input_length,  0, original_idx))
            # print('sorted_idx ', sorted_idx)
            # print('original_idx: ', original_idx)
            # print(output.size())
            return output

        else:
            # 轉為word vector [batch_size, max_len, embed_dim]
            x = self.embed(batch)
            seq_out, (h_n, c_n) = self.rnn(x)
            output = self.classifier(seq_out)
            return output
            # print("output size: ",output.size())
            # print("h_n size: ",h_n.size())


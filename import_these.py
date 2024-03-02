import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

AMINO_ACID = { #{{{
    "G": 0,
    "A": 1,
    "V": 2,
    "L": 3,
    "I": 4,
    "F": 5,
    "W": 6,
    "Y": 7,
    "D": 8,
    "H": 9,
    "N": 10,
    "E": 11,
    "K": 12,
    "Q": 13,
    "M": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "C": 18,
    "P": 19,
    "X": 20
} #}}}


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, max_length=40, main_dir='/home/tiffany/git/iACVP_alldata/data/dataset'):
        self.max_length = max_length
        self.seqs, self.labels = self.load_dataset([f'{main_dir}/{file_path}' for file_path in file_paths])
        self.seqs_token = self.label_encode(self.seqs)

    def load_dataset(self, paths):
        seqs = []
        labels = []
        for path in paths:
            _df = pd.read_csv(path, index_col=None)
            seqs.extend(_df['seq'].to_list())
            labels.extend(_df['label'].to_list())
        assert len(seqs) == len(labels)
        return seqs, labels

    def label_encode(self, seqs):
        seqs_token = []
        for s in seqs:
            s = s.ljust(self.max_length, 'X') if len(s) < self.max_length else s[:self.max_length]
            seqs_token.append([AMINO_ACID[a] for a in s])
        return torch.tensor(seqs_token)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs_token[idx], self.labels[idx]


class Model(nn.Module):
    def __init__(self, hid_dims, num_vocab=len(AMINO_ACID), device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.embedding_layer = nn.Embedding(num_vocab, num_vocab)
        self.embedding_layer.weight = nn.Parameter(torch.eye(num_vocab, num_vocab))
        self.embedding_layer.requires_grad_ = False

        self.conv_layer = nn.Sequential(
            nn.Conv1d(num_vocab, hid_dims[0], kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(hid_dims[0], hid_dims[1], kernel_size=3, stride=1, padding=0),
        )

    def forward(self, x, evaluation=False):
        x = x.to(self.device)         # (b, l)
        seq_len = (x != 20).sum(-1).unsqueeze(-1)  # 20 is X(padding), (b, 1)
        emb = self.embedding_layer(x) # (b, l, c)
        hid_out = self.conv_layer(emb.permute(0, 2, 1)) # (b, c, l)
        if evaluation:
            return hid_out, None

        loss = self.calc_loss(hid_out, seq_len - 4) # 4 is conv no padding
        return None, loss

    def calc_loss(self, hid_out, seq_len, tau=0.1):
        hid_out = hid_out.permute(0, 2, 1)
        b, l, c = hid_out.shape

        # Generate random indices of shape (b, 2) in the range [0, seq_len)
        random_index = (torch.rand(b, 2, device=self.device) * seq_len).long() # (b, 2)
        emb_1 = hid_out[torch.arange(b), random_index[:, 0]] # (b, c)
        emb_2 = hid_out[torch.arange(b), random_index[:, 1]] # (b, c)

        cosine_similarity = torch.cosine_similarity(emb_1.unsqueeze(1), emb_2.unsqueeze(0), dim=-1) # (b,1,c) (1,b,c) -> (b,b)

        loss = torch.nn.functional.cross_entropy(cosine_similarity / tau, torch.arange(b).to(self.device))
        return loss

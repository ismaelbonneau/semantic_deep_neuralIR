print("Importing Libraries")
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import sys
import os
from os import path

libpath = path.normpath(path.join(path.dirname(path.realpath(__file__)), os.pardir))
sys.path.append(libpath)

from datasets import Robust2004
from ir_engine import get_engine, eval_queries

from rl_model import KW_Model, learn, sequence_collate_fn_mask

seed = 69456152

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class UniLSTM(KW_Model):
    def __init__(self, input_dim, hidden_size):
        super(UniLSTM, self).__init__()
        self.embedder = nn.LSTM(input_dim, hidden_size, num_layers=1)
        self.predictor = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def compute_embedding(self, x):
        """
        Returns tensor:
            seq_len x batch x hidden_size
        """
        output, *_ = self.embedder(x)
        return output
    
    def predict_distribution(self, embeddings):
        """
        Returns tensor:
            seq_len x batch
        """
        return self.predictor(embeddings)


device = torch.device("cuda:0")

train_len = 124

dataset = Robust2004.torch_dataset()
dataclasses = Robust2004.dataclasses()
dataclasses = {dc._id: dc for dc in dataclasses}

collate_fn = sequence_collate_fn_mask 
trainset, testset = random_split(dataset, [train_len, len(dataset) - train_len])
trainloader, testloader = DataLoader(trainset, 2, shuffle=True, collate_fn=collate_fn), DataLoader(testset, 64,  collate_fn=collate_fn)

model = UniLSTM(300, 100).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

engine = get_engine()

def eval_fn(preds, qids):
    qids = list(map(lambda x: str(x.long().tolist()), qids))
    maps = {}
    preds = preds.t()
    for pred, qid in zip(preds, qids):
        query = dataclasses[qid].get_text(pred)
        qrel = dataclasses[qid].qrels
        maps[qid] = eval_queries({qid: query}, {qid: qrel}, engine)
    return maps

nb_epoch = 200
model, train_logs, test_logs = learn(model, trainloader, testloader, optimizer, nb_epoch, device, eval_fn)
torch.save(model, "lstm.torchsave")
with open("lstm.txt", "w") as f:
    f.write(str(train_logs))
    f.write("\n" + str(test_logs))

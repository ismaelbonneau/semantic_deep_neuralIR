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

print(__file__)

from datasets import Robust2004
from ir_engine import get_engine, eval_queries
from rl_model import KW_Model, learn, sequence_collate_fn_mask


seed = 5652

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class LinearModel(KW_Model):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.embedder = nn.Sequential(nn.Linear(input_dim, 100), nn.ReLU(), nn.Linear(100, 1))
        self.predictor = nn.Sigmoid()

    def compute_embedding(self, x):
        return self.embedder(x)
    
    def predict_distribution(self, embeddings):
        return self.predictor(embeddings)


device = torch.device("cuda:0")

train_len = 124

dataset = Robust2004.torch_dataset()
dataclasses = Robust2004.dataclasses()
dataclasses = {dc._id: dc for dc in dataclasses}

collate_fn = sequence_collate_fn_mask
trainset, testset = random_split(dataset, [train_len, len(dataset) - train_len])
trainloader, testloader = DataLoader(trainset, 2, shuffle=True, collate_fn=collate_fn), DataLoader(testset, 128,  collate_fn=collate_fn)

model = LinearModel(300).to(device)
optimizer = optim.Adam(model.parameters())

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
torch.save(model, "modelmlp.torchsave")
with open("mlp.txt", "w") as f:
    f.write(str(train_logs))
    f.write("\n" + str(test_logs))

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
from rl_model import learn, sequence_collate_fn_mask, KW_Model
from model import KeyWordSelectionModel_1a, memory_2b


seed = 5652

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


device = torch.device("cuda:0")

train_len = 124

dataset = Robust2004.torch_dataset()
dataclasses = Robust2004.dataclasses()
dataclasses = {dc._id: dc for dc in dataclasses}

collate_fn = sequence_collate_fn_mask
trainset, testset = random_split(dataset, [train_len, len(dataset) - train_len])
trainloader, testloader = DataLoader(trainset, 2, shuffle=True, collate_fn=collate_fn), DataLoader(testset, 128,  collate_fn=collate_fn)


embedding_size = 300
hidden_size = 100
num_layers = 1
bidirectional = True
decoder_archi = {"input_size": embedding_size, "hidden_size": hidden_size, "num_layers": num_layers, "bidirectional":bidirectional, "dropout":0.}
predictor = [2*hidden_size, 1]
model = memory_2b(KeyWordSelectionModel_1a)(decoder_archi, predictor).to(device)
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
with open("1a2b.txt", "w") as f:
    f.write(str(train_logs))
    f.write("\n" + str(test_logs))



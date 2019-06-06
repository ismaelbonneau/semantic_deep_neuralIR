import sys
import os
from os import path

libpath = path.normpath(path.join(path.dirname(path.realpath(__file__)), os.pardir))
sys.path.append(libpath)


print("Load libraries")
from itertools import chain, count
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split, Subset

from datasets import Quora, Robust2004
from data import embedding_collate_decorator, sequence_collate_fn
from model import KeyWordSelectionModel_1c, memory_2b

import random
import numpy as np

seed = 4269666
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# Constant definition
device = torch.device("cuda:1")

# Le probleme vient du count vectorizer qui vire certains mots
print("Load Dataset")
dataset = Quora.torch_dataset()
dataclasses = Quora.dataclasses()
dataclasses = {q._id: q for q in dataclasses}

def embedding_collate_decorator(collate_fn):
    def wrapper(batch):
        x, y, id_, qrels, seq_lens = collate_fn(batch)
        return x, y, id_, qrels, seq_lens
    return wrapper

collate_fn = embedding_collate_decorator(sequence_collate_fn)


train_len, val_len = int(0.7*len(dataset)), int(0.15*len(dataset))
test_len = len(dataset) - train_len - val_len

indices = [(i, i+1) for i in range(0, len(dataset), 2)]
random.shuffle(indices)
indices = list(chain(*indices))
print(len(indices))
if train_len % 2 == 1:
    train_len -= 1
    val_len += 1
if val_len % 2 == 1:
    test_len -= 1
    val_len += 1

assert train_len % 2 == 0
assert val_len % 2 == 0
assert test_len % 2 == 0

trainset, valset, testset = indices[:train_len], indices[train_len:train_len+val_len], indices[train_len+val_len:]


trainset, valset, testset = Subset(dataset, trainset), Subset(dataset, valset), Subset(dataset, testset)
print(len(trainset), len(valset), len(testset))
trainloader = DataLoader(trainset, 1024, True, collate_fn=collate_fn)
valloader = DataLoader(valset, 1024, True, collate_fn=collate_fn)
testloader = DataLoader(testset, 1024, True, collate_fn=collate_fn)

robset = Robust2004.torch_dataset()
rob_loader = DataLoader(robset, 64, True, collate_fn=collate_fn)
rob_dc = Robust2004.dataclasses()
rob_dc = {q._id: q for q in rob_dc}

print("Build model")
embedding_size = 300
hidden_size = 128
num_layers = 3
bidirectional = True

encoder_archi = {"input_size": embedding_size, "hidden_size": hidden_size, "num_layers": num_layers, "bidirectional":True}
decoder_archi = {"input_size": embedding_size + 4*hidden_size, "hidden_size": hidden_size, "num_layers": num_layers, "bidirectional":True}
predictor_structure = [2*hidden_size, 1]

model = memory_2b(KeyWordSelectionModel_1c)(encoder_archi, decoder_archi, predictor_structure)

model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=1e-3)
loss_function = nn.BCELoss()

quora_save = "quora.txt"
robust_save = "robust.txt"


n, mean = 0, 0
for x, y, q_id, qrels, seq_lens in valloader:
    x = x.to(device)
    y = y.to(device)

    pred = model(x)
    loss = loss_function(pred, y.float())
    
    n += 1
    mean = ((n-1) * mean + loss.item()) / n
    print(f"\rInitial Test : {mean}", end="")
print()
n, mean = 0, 0
for x, y, q_id, qrels, seq_lens in rob_loader:
    x = x.to(device)
    y = y.to(device)

    pred = model(x)
    loss = loss_function(pred, y.float())
    
    n += 1
    mean = ((n-1) * mean + loss.item()) / n
    print(f"\rRobust Initial Test : {mean}", end="")
print()
print("Train")
for epoch in count():
    n, mean = 0, 0
    model.train()
    for x, y, q_id, qrels, seq_lens in trainloader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_function(pred, y.float())

        n += 1
        mean = ((n-1) * mean + loss.item()) / n
        print(f"\rEpoch {epoch}\tTrain : {mean}", end="")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
#    model.eval()

    train_mean = mean
    n, mean = 0, 0
    test_predictions = []
    test_ids = []
    for x, y, q_id, qrels, seq_lens in valloader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        pred__ = pred > 0.5
        pred_ = pred__.detach().cpu().long().t().numpy().tolist()
        test_predictions.extend(pred_)
        test_ids.extend(map(lambda x: x.long().tolist(), q_id))

        loss = loss_function(pred, y.float())
        
        n += 1
        mean = ((n-1) * mean + loss.item()) / n
        print(f"\rEpoch {epoch}\tTrain : {train_mean}\tTest : {mean}", end="")
    test_mean = mean
    
    preds = list(zip(test_ids, test_predictions))
    random.shuffle(preds)
    preds = preds[:10]
    
    queries = [(" ".join(dataclasses[str(id_)].text), " ".join(dataclasses[str(id_)].get_text(pred)), " ".join(dataclasses[str(id_)].query)) for id_, pred in preds]
    with open(quora_save, "a") as f:
        f.write(f"\nEpoch {epoch}\n")
        f.write("\n".join(map(str, queries)))

    n, mean = 0, 0
    rob_predictions = []
    rob_ids = []
    for x, y, q_id, qrels, seq_lens in rob_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        pred__ = pred > 0.5
        pred_ = pred__.detach().cpu().long().t().numpy().tolist()
        rob_predictions.extend(pred_)
        rob_ids.extend(map(lambda x: x.long().tolist(), q_id))

        loss = loss_function(pred, y.float())
        
        n += 1
        mean = ((n-1) * mean + loss.item()) / n
        print(f"\rEpoch {epoch}\tTrain : {train_mean}\tTest : {test_mean}\tRob : {mean}", end="")
    print()

    preds = list(zip(test_ids, test_predictions))
    random.shuffle(preds)
    preds = preds[:10]
    
    queries = [(" ".join(dataclasses[str(id_)].text), " ".join(dataclasses[str(id_)].get_text(pred)), " ".join(dataclasses[str(id_)].query)) for id_, pred in preds]
    with open(robust_save, "a") as f:
        f.write(f"\nEpoch {epoch}\n")
        f.write("\n".join(map(str, queries)))

    if epoch % 10 == 0:
        torch.save(model.state_dict(), "save.torchsave")



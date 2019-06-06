import sys
import os
from os import path

libpath = path.normpath(path.join(path.dirname(path.realpath(__file__)), os.pardir))
sys.path.append(libpath)

print("Load libraries")
from itertools import chain
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split, Subset

from datasets import Robust2004
from data import KeyWordSelectionDataset, sequence_collate_fn
from model import KeyWordSelectionModel_1c, memory_2c
from utils import all_but_one
from ir_engine import get_engine, eval_queries

import random
import numpy as np

seed = 666
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# Constant definition
device = torch.device("cuda:2")
embedding_size = 300
hidden_size = 100 
num_layers = 1
bidirectional = True

# Le probleme vient du count vectorizer qui vire certains mots
print("Load Dataset")
dataset = Robust2004.torch_dataset()
dataclasses = Robust2004.dataclasses()
dataclasses = {qt._id: qt for qt in dataclasses}
engine = get_engine()

def embedding_collate_decorator(collate_fn):
    def wrapper(batch):
        x, y, id_, qrel, seq_lens = collate_fn(batch)
        return x, y, id_, qrel, seq_lens
    return wrapper

collate_fn = embedding_collate_decorator(sequence_collate_fn)


indices = list(range(len(dataset)))
random.shuffle(indices)
for i, (trainindices, testindices) in enumerate(all_but_one(indices, k=2)):
    trainindices = chain(*trainindices)
    trainset = Subset(dataset, list(trainindices))
    testset = Subset(dataset, list(testindices))
    trainloader = DataLoader(trainset, 200, True, collate_fn=collate_fn)
    testloader = DataLoader(testset, 200, True, collate_fn=collate_fn)

    print("Build model")
    encoder_archi = {"input_size": embedding_size, "hidden_size": hidden_size, "num_layers": num_layers, "bidirectional":True}
    decoder_archi = {"input_size": embedding_size + 4*hidden_size, "hidden_size": hidden_size, "num_layers": num_layers, "bidirectional":True}
    predictor_structure = [2*hidden_size, 1]
    model = memory_2c(KeyWordSelectionModel_1c)(encoder_archi, decoder_archi, predictor_structure)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters())
    loss_function = nn.BCELoss()

    print("Train")
    best_model = -1
    delay = 0
    max_delay = 10
    for epoch in range(500):
        model.train()
        n, mean = 0, 0
        train_predictions = []
        train_ids = []
        for x, y, q_id, qrels, seq_lens in trainloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            
            pred__ = pred > 0.5
            pred_ = pred__.detach().cpu().long().t().numpy().tolist()
            train_predictions.extend(pred_)
            train_ids.extend(map(lambda x: x.long().tolist(), q_id))

            loss = loss_function(pred, y.float())

            n += 1
            mean = ((n-1) * mean + loss.item()) / n
            print(f"\rFold {i}, Epoch {epoch}\tTrain : {mean}", end="")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
       
        train_queries = {id_: dataclasses[str(id_)].get_text(pred) for id_, pred in zip(train_ids, train_predictions)}
        train_qrel = {id_: dataclasses[str(id_)].qrels for id_, pred in zip(train_ids, train_predictions)}
        train_map = eval_queries(train_queries, train_qrel, engine) 
        print(f"\rFold {i}, Epoch {epoch}\tTrain Loss: {mean}, Train MAP {train_map}", end="")
        model.eval()

        train_mean = mean
        n, mean = 0, 0
        test_predictions = []
        test_ids = []
        for x, y, q_id, qrels, seq_lens in testloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            pre__ = pred > 0.5
            pred_ = pred__.detach().cpu().long().t().numpy().tolist()
            test_predictions.extend(pred_)
            test_ids.extend(map(lambda x: x.long().tolist(), q_id))

            loss = loss_function(pred, y.float())

            n += 1
            mean = ((n-1) * mean + loss.item()) / n
            print(f"\rFold {i}, Epoch {epoch}\tTrain : {train_mean}\tTest : {mean}", end="")
        print()

        test_queries = {id_: dataclasses[str(id_)].get_text(pred) for id_, pred in zip(test_ids, test_predictions)}
        test_qrel = {id_: dataclasses[str(id_)].qrels for id_, pred in zip(test_ids, test_predictions)}
        test_map = eval_queries(test_queries, test_qrel, engine)
        
        dataset_queries = {**train_queries, **test_queries}
        dataset_qrel = {**train_qrel, **test_qrel}
        dataset_map = eval_queries(dataset_queries, dataset_qrel, engine)

        print("\b"*500 + f"\nFold {i}, Epoch {epoch}\tTrain MAP {train_map}\tTest MAP : {test_map}\tDataset MAP : {dataset_map}")
       

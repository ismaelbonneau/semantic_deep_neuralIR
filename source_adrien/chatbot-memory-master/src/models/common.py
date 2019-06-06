import sys
import os
from os import path

libpath = path.normpath(path.join(path.dirname(path.realpath(__file__)), os.pardir))
sys.path.append(libpath)

print("Load libraries")
import shutil
import random
from itertools import chain, islice, repeat

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli

from torch.utils.data import DataLoader, random_split, Subset

from datasets import Robust2004
from data import KeyWordSelectionDataset, sequence_collate_fn, embedding_collate_decorator
from utils import all_but_one
from ir_engine import get_engine, eval_queries
from model import KeyWordSelectionModel_1c, memory_2c


# Constant definition

def learn(model, model_args, device, k=5, batch_size=32, seed=666, smt_epoch=100, rl_epoch=1000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Le probleme vient du count vectorizer qui vire certains mots
    print("Load Dataset")
    dataset = Robust2004.torch_dataset()
    dataclasses = Robust2004.dataclasses()
    dataclasses = {qt._id: qt for qt in dataclasses}
    engine = get_engine()

    collate_fn = embedding_collate_decorator(sequence_collate_fn)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    for i, (trainindices, testindices) in enumerate(all_but_one(indices, k=k)):
        trainindices = chain(*trainindices)
        trainset = Subset(dataset, list(trainindices))
        testset = Subset(dataset, list(testindices))
        trainloader = DataLoader(trainset, 1, True, collate_fn=collate_fn)
        testloader = DataLoader(testset, 1, True, collate_fn=collate_fn)

        print("Build model")

        model = model(*model_args)
        try:
            model = model.to(device)
        except RuntimeError:
            print("cudnn error")
        model = model.to(device)


        optimizer = optim.Adam(model.parameters())
        loss_function = nn.BCELoss()

        print("Train")
        best_model = 0
        delay = 0
        max_delay = 5
        print("Supervised Machine Translation")
        for epoch in range(smt_epoch):
            model.train()
            n, mean = 0, 0
            train_predictions = []
            train_ids = []
            for x, y, q_id, qrels, _ in trainloader:
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
            for x, y, q_id, qrels, _ in testloader:
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
                print(f"\rFold {i}, Epoch {epoch}\tTrain Loss: {train_mean}\tTest : {mean}", end="")
            
            test_queries = {id_: dataclasses[str(id_)].get_text(pred) for id_, pred in zip(test_ids, test_predictions)}
            test_qrel = {id_: dataclasses[str(id_)].qrels for id_, pred in zip(test_ids, test_predictions)}
            test_map = eval_queries(test_queries, test_qrel, engine)
            
            dataset_queries = {**train_queries, **test_queries}
            dataset_qrel = {**train_qrel, **test_qrel}
            dataset_map = eval_queries(dataset_queries, dataset_qrel, engine)

            print("\b"*500 + f"\nFold {i}, Epoch {epoch}\tTrain MAP {train_map}\tTest MAP : {test_map}\tDataset MAP : {dataset_map}")
            
            if test_map > best_model:
                best_model = test_map
                delay = 0
            elif test_map < best_model:
                delay += 1
                if delay > max_delay:
                    print(best_model)
                    break
        
        print("Reinforcement Learning")
        mean_maps = {id_: [] for id_ in dataclasses.keys()}
        for epoch in range(rl_epoch):
            model.train()
            n, mean = 0, 0
            train_predictions = []
            train_ids = []
            for x, y, q_id, qrels, seq_lens in trainloader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)

                sampler = Bernoulli(pred)
                
                batch_pred = sampler.sample()
                log_probs = sampler.log_prob(batch_pred)
                loss = log_probs.sum()

                batch_ids = list(map(lambda x: x.long().tolist(), q_id))

                batch_queries = {id_: dataclasses[str(id_)].get_text(pred) for id_, pred in zip(batch_ids, batch_pred)}
                batch_qrel = {id_: dataclasses[str(id_)].qrels for id_, pred in zip(batch_ids, batch_pred)}
                
                batch_map = eval_queries(batch_queries, batch_qrel, engine)
                
                n += 1
                mean = ((n-1) * mean + batch_map) / n
                print(f"\rTrain Map : {mean: .3f}", end="")
                loss = -batch_map * loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_mean = mean
            n, mean = 0, 0
            test_predictions = []
            test_ids = []
            print()
            for x, y, q_id, qrels, seq_lens in testloader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)

                sampler = Bernoulli(pred)
                batch_pred = sampler.sample()
                log_probs = sampler.log_prob(batch_pred)
                loss = log_probs.sum()
                batch_qrel = {id_: dataclasses[str(id_)].qrels for id_, pred in zip(batch_ids, batch_pred)}
                
                batch_map = eval_queries(batch_queries, batch_qrel, engine)
                n += 1
                mean = ((n-1) * mean + batch_map) / n
                print(f"\rTrain MAP : {train_mean: .3f}\tTest Map : {mean: .3f}", end="")
            print()


embedding_size = 300
hidden_size = 100 
num_layers = 1
bidirectional = False

decoder_archi = {"input_size": embedding_size, "hidden_size": hidden_size, "num_layers": num_layers, "bidirectional":bidirectional}
predictor = [hidden_size, 1]
device = torch.device("cuda:0")
model = memory_2c(KeyWordSelectionModel_1c)

learn(model, [decoder_archi, predictor], device, k=2, smt_epoch=20)

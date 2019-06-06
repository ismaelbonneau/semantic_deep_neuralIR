import shutil
import random
from os import path
from itertools import chain, islice, repeat, count

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader, random_split, Subset

from utils import all_but_one
from data import embedding_collate_decorator, sequence_collate_fn


def cross_val(dataset, nb_fold, batch_size, collate_fn):
    """Given a dataset yields data for cross-validation.

    Parameters
    -----------
        dataset : torch.Dataset
            The dataset used
        
        nb_fold : int
            Number of fold used for cross-validation
        
        batch_size : int
            The batch size
    
    Yields
    ------
        trainloader : torch.Dataloader
        testloader : torch.Dataloader
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for trainindices, testindices in all_but_one(indices, k=nb_fold):
        trainindices = chain(*trainindices)
        trainset = Subset(dataset, list(trainindices))
        testset = Subset(dataset, list(testindices))
        trainloader = DataLoader(trainset, batch_size, shuffle=True, collate_fn=collate_fn)
        testloader = DataLoader(testset, batch_size, shuffle=True, collate_fn=collate_fn)

        yield trainloader, testloader

def smt_learning(model, criterion, optimizer, trainloader, testloader, 
                 device, nb_epoch=None, max_delay=3, save_best_every=None, save_dir="."):
    """Supervised Machine Translation learning loop.

    Parameters
    -----------
        model       : torch.nn.Module        
        criterion   : torch.nn loss function
        optimizer   : torch.optim optimizer        
        trainloader : torch.utils.data.DataLoader
        testloader  : torch.utils.data.DataLoader
        device      : torch.device
        nb_epoch    : int or None, default None
            If None, use itertools.count()
        max_delay : int, default 3
            Delay before leaving the learning if there's no improvement
            on the test loss. If None, there's no delay
        save_best_every : int or None, default None
            Frequency at which the best model is saved. If None,
            never save it.
        save_dir : str or path

    Returns
    --------
        model : torch.nn.Module
        best_model : torch.nn.Module.state_dict
    """
    if save_best_every is None:
        save_best_every = nb_epoch
    
    if nb_epoch is None:
        counter = count(1)
    else:
        counter = range(1, nb_epoch+1)

    best_score = 0
    best_model = dict()
    delay = 0
    for epoch in counter:
        model.train()
        n, mean = 0, 0
        with tqdm(trainloader, ncols=80, desc=f"Epoch {epoch}") as t:
            for x, y, *_ in t:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = criterion(pred, y.float())

                n += 1
                mean = ((n-1) * mean + loss.item()) / n
                t.set_postfix({"Train": "{0:.3f}".format(mean), "Test": "NaN"})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss = mean
            model.eval()
            n, mean = 0, 0
            for x, y, *_ in testloader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = criterion(pred, y.float())
                
                n += 1
                mean = ((n-1) * mean + loss.item()) / n

            test_loss = mean
            t.set_postfix({"Train": "{0:.3f}".format(train_loss), "Test": "{0:.3f}".format(mean)})

        if epoch % save_best_every == 0:
            torch.save(best_model, path.join(save_dir, "best.torchsave"))

        if test_loss < best_score:
            best_score = test_loss
            best_model = model.state_dict().copy()
            delay = 0
        elif test_loss > best_score:
            if test_loss < train_loss:
                delay += 1
                if delay > max_delay:
                    print(f"Best Score : {best_score}")
                    break
    return model, best_model


def rl_learning(model, eval_fn, optimizer, trainloader, testloader, dataclasses, 
                 device, nb_epoch=None, max_delay=3, save_best_every=None, save_dir="."):
    """Reinforcement Learning learning loop.

    Parameters
    -----------
        model       : torch.nn.Module        
        eval_fn     : functino
            The function called to get the reward.
            Used like : eval_fn(queries, qrels).
            "queries" and "qrels" are dict like : {id_q: text} and {id_q: qrel}
        optimizer   : torch.optim optimizer        
        trainloader : torch.utils.data.DataLoader
        testloader  : torch.utils.data.DataLoader
        dataclasses : Dict[data.QueryText]
        device      : torch.device
        nb_epoch    : int or None, default None
            If None, use itertools.count()
        max_delay : int, default 3
            Delay before leaving the learning if there's no improvement
            on the test loss. If None, there's no delay
        save_best_every : int or None, default None
            Frequency at which the best model is saved. If None,
            never save it.
        save_dir : str or path

    Returns
    --------
        model : torch.nn.Module
        best_model : torch.nn.Module.state_dict
    """
    if save_best_every is None:
        save_best_every = nb_epoch
    
    if nb_epoch is None:
        counter = count(1)
    else:
        counter = range(1, nb_epoch+1)

    best_map = 0
    best_model = dict()
    for epoch in counter():
        model.train()
        n, mean = 0, 0
        for x, y, q_id in trainloader:
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
            
            batch_map = eval_fn(batch_queries, batch_qrel)
            
            n += 1
            mean = ((n-1) * mean + batch_map) / n
            print(f"\rTrain Map : {mean}", end="")
            loss = -batch_map * log_probs
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        train_map = mean
        n, mean = 0, 0
        for x, y, q_id in testloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            sampler = Bernoulli(pred)
            batch_pred = sampler.sample()
            log_probs = sampler.log_prob(batch_pred)
            loss = log_probs.sum()
            batch_qrel = {id_: dataclasses[str(id_)].qrels for id_, pred in zip(batch_ids, batch_pred)}
            
            batch_map = eval_fn(batch_queries, batch_qrel)

            n += 1
            mean = ((n-1) * mean + batch_map) / n
            print(f"\rtrain Map : {train_map}\tTest Map : {batch_map}", end="")
        print()
        test_map = mean

        if epoch % save_best_every == 0:
            torch.save(best_model, path.join(save_dir, "best.torchsave"))

        if test_map > best_score:
            best_score = test_map
            best_model = model.state_dict().copy()
            delay = 0
        elif test_map < best_score:
            if test_map > train_map:
                delay += 1
                if delay > max_delay:
                    print(f"Best Score : {best_score}")
                    break
    return best_model
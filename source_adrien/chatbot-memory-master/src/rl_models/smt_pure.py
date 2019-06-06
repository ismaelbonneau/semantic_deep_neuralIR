import sys
import os
from os import path

libpath = path.normpath(path.join(path.dirname(path.realpath(__file__)), os.pardir))
sys.path.append(libpath)

from typing import Dict, Callable, List, NewType
from collections import deque
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader

Qid = NewType("Qid", str)


def mean(iterable):
    if len(iterable) > 0:
        return sum(iterable) / len(iterable)
    return 0


# TODO : add mask for entropy/logit computation

def learn(model: nn.Module,
          trainloader: DataLoader,
          testloader: DataLoader,
          optimizer: optim.Optimizer,
          nb_epoch: int,
          device: torch.device,
          eval_fn: Callable[[List[bool], List[Qid]], Dict[Qid, float]],
          mean_window: int = 50,
          entropy_lambda: float = 0.,
          ) -> str:
    """Just fuckin learn. Please."""
    
    past_rewards = {str(q_id.long().tolist()): deque(maxlen=mean_window)
                        for _, _, q_ids, _, _ in chain(trainloader, testloader)
                        for q_id in q_ids}
    print("Learning")
    l = 1
    for i in range(nb_epoch):
        running_loss = []
        running_reward = []
        for x, y, q_id, _, _ in trainloader:
            x = x.to(device)
            y = y.to(device)

            # batch x seq , batch x seq , batch x seq
            sample, logits, entropy, params = model(x)

            #########
            # Reinforce
            #########            
            rewards = eval_fn(sample, q_id)
            advantage_reward = [r - mean(past_rewards[q_id])
                                    for q_id, r in rewards.items()]
           
            advantage_reward = torch.FloatTensor(advantage_reward).unsqueeze(1).to(device)
            advantage_reward *= 1
            
            # seq x batch * batch x 1
            reinforce_loss = (logits.t() * advantage_reward).sum(0).mean()

            smt_loss = nn.BCELoss()(params, y.float())
            #########
            # Loss
            #########
            entropy = -entropy.mean()
            loss = 0*reinforce_loss + entropy_lambda * entropy + l*smt_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            running_reward.append(mean(rewards.values()))
            print(f"\rTr Loss {mean(running_loss): .3f} Rewa {mean(running_reward): .5f}, {reinforce_loss: .5f}, {entropy*entropy_lambda: .3f}, {smt_loss*l: .3f}", end="")
        print()
        train_loss, train_reward = mean(running_loss), mean(running_reward)
        running_loss = []
        running_reward = []
        for x, y, q_id, _, _ in trainloader:
            x = x.to(device)
            y = y.to(device)

            # batch x seq , batch x seq , batch x seq
            sample, logits, entropy, params = model(x)

            #########
            # Reinforce
            #########            
            rewards = eval_fn(sample, q_id)
            advantage_reward = [r - mean(past_rewards[q_id])
                                    for q_id, r in rewards.items()]
            advantage_reward = torch.FloatTensor(advantage_reward).unsqueeze(1).to(device)
            
            # batch x seq * batch x 1
            reinforce_loss = (logits.t() * advantage_reward).sum(0).mean()

            ### SMT
            smt_loss = nn.BCELoss()(params, y.float())
            #########
            # Loss
            #########
            loss = reinforce_loss + entropy_lambda * entropy.mean()

            running_loss.append(loss.item())
            running_reward.append(mean(rewards.values()))

            print(f"\rTr Loss {train_loss: .3f} Rewa {train_reward: .3f}\t"
                  + f"Te Loss {mean(running_loss): .3f} Rewa {mean(running_reward): .3f}", end="")
        print()
        with open("smt.txt", "a") as f:
            f.write(f"{train_loss} {train_reward} {mean(running_loss)} {mean(running_reward)}\n")
    return


if __name__ == "__main__":
    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()
            self.net = nn.Sequential(nn.Linear(300, 1),  nn.Sigmoid())
        
        def forward(self, x):
            seq, batch = x.size(0), x.size(1)

            x = x.view(batch*seq, -1)
            params = self.net(x)
            params = params.view(seq, batch)
            
            sampler = Bernoulli(params)
            pred = sampler.sample()

            logits = sampler.log_prob(pred)
            entropy = sampler.entropy().sum(0)

            return pred, logits, entropy, params
    
    from functools import partial
    from torch.utils.data import random_split
    from datasets import Robust2004
    from ir_engine import get_engine, eval_queries
    from data import embedding_collate_decorator, sequence_collate_fn
    
    import numpy as np
    import random

    seed = 69456152

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    collate_fn = embedding_collate_decorator(sequence_collate_fn)
    model = Test().to(torch.device("cuda:1"))
    dataset = Robust2004.torch_dataset()
    dataclasses = Robust2004.dataclasses()
    dataclasses = {dc._id: dc for dc in dataclasses}
    trainset, testset = random_split(dataset, [125, len(dataset) - 125])
    trainloader, testloader = DataLoader(trainset, 16, shuffle=True, collate_fn=collate_fn), DataLoader(testset, 64,  collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters())
    nb_epoch = 2000

    engine = get_engine()

    def toto(preds, qids):
        qids = list(map(lambda x: str(x.long().tolist()), qids))
        maps = {}
        preds = preds.t()
        for pred, qid in zip(preds, qids):
            query = dataclasses[qid].get_text(pred)
            qrel = dataclasses[qid].qrels
            maps[qid] = eval_queries({qid: query}, {qid: qrel}, engine)
        return maps

    learn(model, trainloader, testloader, optimizer, nb_epoch, torch.device("cuda:1"), toto)


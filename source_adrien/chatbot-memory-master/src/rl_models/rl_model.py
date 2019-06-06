import gc
import collections
import sys
import resource
import psutil 

import abc
from itertools import chain
from collections import deque
from typing import Callable, List, NewType, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


Qid = NewType("Qid", str)

def apply_mask(tensor, mask):
    return tensor.squeeze() * mask


class KW_Model(nn.Module):
    def __init__(self):
        super(KW_Model, self).__init__()
    
    def forward(self, x, mask):
        """
            x.size() = ("seq", "batch", ...)
        """
        # Compute the parameters for the Bernoulli
        embeddings = self.compute_embedding(x)
        dist_params = self.predict_distribution(embeddings)
        dist_params = dist_params.squeeze() 
        # Sample
        sampler = dist.Bernoulli(probs=dist_params)
        actions = sampler.sample()
        
        # Compute LogProba
        log_probas = sampler.log_prob(actions)
        log_probas = apply_mask(log_probas, mask)

        # Compute Entropy
        entropy = sampler.entropy()
        entropy = apply_mask(log_probas, mask)
        
        return actions, log_probas, entropy, dist_params

    @abc.abstractmethod
    def compute_embedding(self, x):
        """
        Returns tensor:
            seq_len x batch x hidden_size
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def predict_distribution(self, embeddings):
        """
        Returns tensor:
            seq_len x batch
        """
        raise NotImplementedError


def mean(iterable):
    if len(iterable) > 0:
        return sum(iterable) / len(iterable)
    return 0


def reinforce_adv(rewards, logits, past_rewards, device):
    advantage_reward = [r - mean(past_rewards[q_id])
                        for q_id, r in rewards.items()]
    advantage_reward = torch.FloatTensor(advantage_reward).unsqueeze(1).to(device)
    
    # seq x batch * batch x 1
    return (logits.t() * advantage_reward).sum(0).mean()

def compute_losses(y, 
                   dist_params,
                   rewards,
                   logits,
                   past_rewards,
                   entropy,
                   entropy_lambda,
                   reinforce_lambda,
                   smt_lambda,
                   device):
    #########
    # Entropy
    #########
    entropy = entropy.mean()
    entropy *= -entropy_lambda

    #########
    # Reinforce
    #########
    reinforce_loss = reinforce_adv(rewards, logits, past_rewards, device)
    reinforce_loss *= reinforce_lambda

    #########
    # SMT
    #########
    smt_loss = nn.BCELoss()(dist_params, y.float())
    smt_loss *= smt_lambda

    #########
    # Loss
    #########
    loss = -reinforce_loss + entropy + smt_loss

    return loss, reinforce_loss, -entropy, smt_loss


def learn(model: KW_Model,
          trainloader: DataLoader,
          testloader: DataLoader,
          optimizer: optim.Optimizer,
          nb_epoch: int,
          device: torch.device,
          eval_fn: Callable[[List[bool], List[Qid]], Dict[Qid, float]],
          mean_window: int = 50,
          entropy_lambda: float = 0.025,
          smt_lambda: float = 1.0,
          reinforce_lambda: float = 1.0,
          ) -> Tuple[nn.Module, Dict[str, List[torch.tensor]], Dict[str, List[torch.tensor]]]:
    """Just fuckin learn. Please."""
    print("Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    past_rewards = {str(q_id.long().item()): deque(maxlen=mean_window)
                    for _, _, q_ids, _ in chain(trainloader, testloader)
                    for q_id in q_ids}
    
    logs = ["reward",
            "scaled_entropy",
            "scaled_reinforce",
            "scaled_smt",
            "total_loss",
            "accuracy"]
    train_logs = {log: list() for log in logs}
    test_logs = {log: list() for log in logs}
    del logs
   
    for epoch in range(nb_epoch):
        running_loss, running_reward = [], []
        entropies, reinforces, smts = [], [], []
        nb_correct, nb_total = 0, 0
        print(f"\nEpoch {epoch}")
        
        print("Begin epoch: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        model.train()
        for x, y, q_id, masks in trainloader:
            x = x.to(device)
            y = y.to(device)
            masks = masks.to(device)

            # batch x seq , batch x seq , batch x seq
            sample, logits, entropy, params = model(x, masks)
            batch_reward, rewards = eval_fn(sample.detach().t().tolist(), q_id)
            losses = compute_losses(y, params, rewards, logits, past_rewards,
                                    entropy, entropy_lambda, reinforce_lambda, smt_lambda, device)

            # entropy_lambda = min(1.01*entropy_lambda, 0.025)
            # reinforce_lambda = min(1.01*reinforce_lambda, 1.0)
            # smt_lambda = max(0.99*smt_lambda, 0.05)
            loss, reinforce_loss, entropy, smt_loss = losses
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            temp = (sample.long() == y.detach().long()).float() * masks
            nb_correct += temp.byte().cpu().sum().tolist()
            nb_total += masks.cpu().sum().tolist()

            running_loss.append(loss.item())
            running_reward.extend(rewards.values())
            print(f"\rTr Loss {mean(running_loss): .3f} Rewa {mean(running_reward): .5f}", end="")
                    
            reinforces.append(reinforce_loss.item())
            entropies.append(entropy.item())
            smts.append(smt_loss.item())

        # Logs
        train_logs["reward"].append(mean(running_reward))
        train_logs["scaled_entropy"].append(mean(entropies))
        train_logs["scaled_reinforce"].append(mean(reinforces))
        train_logs["scaled_smt"].append(mean(smts))
        train_logs["total_loss"].append(mean(running_loss))
        train_logs["accuracy"].append(nb_correct / nb_total)
        
        
        train_loss, train_reward = mean(running_loss), mean(running_reward)
        running_loss, running_reward = [], []
        entropies, reinforces, smts = [], [], []
        nb_correct, nb_total = 0, 0
        model.eval()
        for x, y, q_id, masks in testloader:
            x = x.to(device)
            y = y.to(device)
            masks = masks.to(device)

            # batch x seq , batch x seq , batch x seq
            sample, logits, entropy, params = model(x, masks)
            batch_reward, rewards = eval_fn(sample.detach().t().tolist(), q_id)

            losses = compute_losses(y, params, rewards, logits, past_rewards,
                                    entropy, entropy_lambda, reinforce_lambda, smt_lambda, device)
            loss, reinforce_loss, entropy, smt_loss = losses
            
            temp = (sample.long() == y.detach().long()).float() * masks
            nb_correct += temp.byte().sum().tolist()
            nb_total += masks.sum().tolist()

            running_loss.append(loss.item())
            running_reward.extend(rewards.values())
            print(f"\rTr Loss {train_loss: .3f} Rewa {train_reward: .3f}",
                  f"Te Loss{mean(running_loss): .3f} Rewa {mean(running_reward): .3f}",
                  end="")
            
            reinforces.append(reinforce_loss.item())
            entropies.append(entropy.item())
            smts.append(smt_loss.item())
    
         
        # Logs
        test_logs["reward"].append(mean(running_reward))
        test_logs["scaled_entropy"].append(mean(entropies))
        test_logs["scaled_reinforce"].append(mean(reinforces))
        test_logs["scaled_smt"].append(mean(smts))
        test_logs["total_loss"].append(mean(running_loss))
        test_logs["accuracy"].append(nb_correct / nb_total)
        

    return model, train_logs, test_logs


def sequence_collate_fn_mask(batch):
    x, y, ids, _ = zip(*batch)
    x, y, ids = zip(*sorted(zip(x, y, ids), key=lambda item: item[0].size(0), reverse=True))
    
    masks = [torch.ones(a.size(0)) for a in x]
    masks = pad_sequence(masks)
    x = pad_sequence(x)
    y = pad_sequence(y)
    return x, y, ids, masks

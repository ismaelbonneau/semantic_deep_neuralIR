import sys
import os
from os import path
#libpath = path.normpath(path.join(path.dirname(path.realpath(__file__)), os.pardir))
#sys.path.append(libpath)

sys.path.append("/local/pouyet/py37/chatbot-memory/src/")
sys.path.append("/local/pouyet/py37/chatbot-memory/src/rl_models")
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument("-d", "--device", type=int, required=True)
parser.add_argument("-rl", "--reinforce_lambda", type=float, default=1.0)
parser.add_argument("-smt", "--smt_lambda", type=float, default=1.0)
parser.add_argument("-ent", "--entropy_lambda", type=float, default=0.025)
parser.add_argument('-b', "--batch", type=int, default=2)
parser.add_argument('-mh', "--heads", type=int, default=8)
parser.add_argument("-l", "--nlayers", type=int, default=1)
parser.add_argument("-hs", "--hsize", type=int, default=100)
parser.add_argument("-i", "--inner", type=int, default=100)
parser.add_argument("-n", "--nb_epoch", type=int, default=400)
parser.add_argument("--seed", type=int, default=5652)
 
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from datasets import Robust2004
from ir_engine import get_engine, eval_queries
from rl_model import learn, sequence_collate_fn_mask, KW_Model
from attention import AttModel


from cross_val import cross_val


import resource

qrel_file_path = "/local/pouyet/datasets/as_projet/querries/robust2004_qrel.txt"

def eval_fn(preds, qids, dataclasses, engine, resp_filename):
    qids = list(map(lambda x: str(x.long().tolist()), qids))
    queries = {qid: dataclasses[qid].get_text(pred) for pred, qid in zip(preds, qids)}
    total_score, scores = eval_queries(queries, qrel_file_path, engine, resp_filename)
    return total_score, scores

def main():
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print("Building dataset")
    dataset = Robust2004.torch_dataset()
    dataclasses = Robust2004.dataclasses()
    dataclasses = {dc._id: dc for dc in dataclasses}

    collate_fn = sequence_collate_fn_mask
    trainlogs, testlogs = {}, {}
    for i, (trainloader, testloader) in enumerate(cross_val(dataset, 5, args.batch, collate_fn)):
        
        model = AttModel(args.heads, 300, args.inner, args.hsize, args.hsize, args.nlayers, 0.1)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(pytorch_total_params, "parameters")

        device = torch.device("cuda:" + str(args.device))
        try:
            model = model.to(device)
        except:
            model = model.to(device)
        
        optimizer = optim.Adam(model.parameters())
        print("Getting Engine")
        engine = get_engine(hosts=["localhost:9200"])
    
        resp_filename = f"transformer-{args.smt_lambda}-{args.reinforce_lambda}-{args.entropy_lambda}_{args.batch}-{args.nlayers}-{args.hsize}.resp"
        initialized_eval_fn = partial(eval_fn, dataclasses=dataclasses, engine=engine, resp_filename=resp_filename)
        print("Training")
        model, train_logs, test_logs = learn(
                                    model,
                                    trainloader,
                                    testloader,
                                    optimizer,
                                    args.nb_epoch,
                                    device,
                                    initialized_eval_fn,
                                    50,
                                    args.entropy_lambda,
                                    args.smt_lambda,
                                    args.reinforce_lambda
                                )
        trainlogs[i] = train_logs
        testlogs[i] = test_logs
    with open(f"transformer-{args.smt_lambda}-{args.reinforce_lambda}-{args.entropy_lambda}_{args.batch}-{args.nlayers}-{args.hsize}.txt", "w") as f:
        f.write(str(trainlogs))
        f.write("\n"+ str(testlogs))


if __name__ == "__main__":
    main()

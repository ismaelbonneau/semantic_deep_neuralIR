import sys
import os
from os import path
#libpath = path.normpath(path.join(path.dirname(path.realpath(__file__)), os.pardir))
#sys.path.append(libpath)

sys.path.append("/local/pouyet/py37/chatbot-memory/src/")
sys.path.append("/local/pouyet/py37/chatbot-memory/src/rl_models")
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument("model", choices=["a", "b", "c"])
parser.add_argument("memory", choices=["a", "b", "c"])
parser.add_argument("-d", "--device", type=int, required=True)
parser.add_argument("-rl", "--reinforce_lambda", type=float, default=1.0)
parser.add_argument("-smt", "--smt_lambda", type=float, default=1.0)
parser.add_argument("-ent", "--entropy_lambda", type=float, default=0.025)
parser.add_argument('-b', "--batch", type=int, default=2)
parser.add_argument("-l", "--nlayers", type=int, default=1)
parser.add_argument("-hs", "--hsize", type=int, default=100)
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
from model import *


from cross_val import cross_val


import resource

qrel_file_path = "/local/pouyet/datasets/as_projet/querries/robust2004_qrel.txt"

def eval_fn(preds, qids, dataclasses, engine, resp_filename):
    qids = list(map(lambda x: str(x.long().tolist()), qids))
#    preds = preds.t()

    queries = {qid: dataclasses[qid].get_text(pred) for pred, qid in zip(preds, qids)}
    total_score, scores = eval_queries(queries, qrel_file_path, engine, resp_filename)
    return total_score, scores

def main():
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print("Building dataset")
#    train_len = 200
    dataset = Robust2004.torch_dataset()
    dataclasses = Robust2004.dataclasses()
    dataclasses = {dc._id: dc for dc in dataclasses}

    collate_fn = sequence_collate_fn_mask
    trainlogs, testlogs = {}, {}
    for i, (trainloader, testloader) in enumerate(cross_val(dataset, 5, args.batch, collate_fn)):
        model = eval("KeyWordSelectionModel_1" + args.model)
        memory = eval("memory_2" + args.memory)
        archi_function = eval(args.model + "_archi")
        model1= memory(model)(*archi_function(args.nlayers, args.hsize))
        
        device = torch.device("cuda:" + str(args.device))
        model1 = model1.to(device)
        
        model = model1
        optimizer = optim.Adam(model.parameters())
    #    optimizer = optim.SGD(model.parameters(), lr=1e-3)
        print("Getting Engine")
        engine = get_engine(hosts=["localhost:9200"])
    
        resp_filename = f"{args.model}{args.memory}-{args.smt_lambda}-{args.reinforce_lambda}-{args.entropy_lambda}_{args.batch}-{args.nlayers}-{args.hsize}.resp"
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
    with open(f"{args.model}{args.memory}-{args.smt_lambda}-{args.reinforce_lambda}-{args.entropy_lambda}_{args.batch}-{args.nlayers}-{args.hsize}.txt", "w") as f:
        f.write(str(trainlogs))
        f.write("\n"+ str(testlogs))
    

def a_archi(nlayers, hsize):
    embedding_size = 300
    hidden_size = hsize
    num_layers = nlayers
    bidirectional = True
    decoder_archi = {"input_size": embedding_size, "hidden_size": hidden_size, "num_layers": num_layers, "bidirectional":bidirectional, "dropout":0.2}
    predictor = [2*hidden_size, 1]

    return decoder_archi, predictor

def b_archi(nlayers, hsize):
    embedding_size = 300
    hidden_size = hsize
    num_layers = nlayers
    encoder_archi = {"input_size": embedding_size, "hidden_size": hidden_size,
                     "num_layers": num_layers, "bidirectional":True, "dropout":0.2}
    decoder_archi = {"input_size": embedding_size, "hidden_size": hidden_size,
                     "num_layers": num_layers, "bidirectional":True, "dropout":0.2}
    predictor_structure = [2*hidden_size, 1]

    return encoder_archi, decoder_archi, predictor_structure

def c_archi(nlayers, hsize):
    embedding_size = 300
    hidden_size = hsize
    num_layers = nlayers
    encoder_archi = {"input_size": embedding_size, "hidden_size": hidden_size, 
                     "num_layers": num_layers, "bidirectional":True, "dropout":0.2}
    decoder_archi = {"input_size": embedding_size + 4*hidden_size, "hidden_size": hidden_size,
                     "num_layers": num_layers, "bidirectional":True, "dropout":0.2}
    predictor_structure = [2*hidden_size, 1]

    return encoder_archi, decoder_archi, predictor_structure

if __name__ == "__main__":
    print("Toto")
    main()

import os.path as path

from itertools import starmap
from dataclasses import dataclass

import numpy as np
import pickle
import torch
import sklearn.feature_extraction.text as text
import fastText

from pytrec_eval import parse_qrel

import sys
import os
from os import path

libpath = path.normpath(path.join(path.dirname(path.realpath(__file__)), os.pardir, "src"))
sys.path.append(libpath)

from data import KeyWordSelectionDataset, QueryText

datasets_path = path.expandvars("$DATASET_HOME/as_projet/querries")
dataset_path = path.join(datasets_path, "robust2004.txt")
qrels_path = path.join(datasets_path, "robust2004_qrel.txt")
dataset_classes_path = path.join(datasets_path, "robust2004_dataclasses.pickled")
torchdataset_path = path.join(datasets_path, "robust2004.torchdataset")

def strips(dataset):
    for id_, (query, text) in dataset.items():
        query = query.split(" ")
        text = text.split(" ")
        id_ = id_
        yield id_, query, text


# Read dataset
with open(dataset_path, "r") as dataset_file:
    dataset = eval(dataset_file.read())

# Transform dataset and get relevent words
dataset_class = list(starmap(QueryText, strips(dataset)))

with open(qrels_path, "r") as f:
    qrels = parse_qrel(f)

# Prepare embeddings
model = fastText.load_model("/local/pouyet/py37/models/wiki.en.bin")
for x in dataset_class:
    x.compute_embedding(model)
    x.qrels = qrels[str(x._id)]

with open(dataset_classes_path, "wb") as f:
    pickle.dump(dataset_class, f)

# Build torch dataset
dataset_torch = KeyWordSelectionDataset(querytext_list=dataset_class)
torch.save(dataset_torch, torchdataset_path)


import gc

import re
import pickle as pkl
import os.path as path
from dataclasses import dataclass, field
from typing import List

import torch
import pandas as pd
from nltk.corpus import stopwords
from torch.utils.data import Dataset, DataLoader
import fastText

from contractions import expand_contractions

import sys
import os
from os import path

libpath = path.normpath(path.join(path.dirname(path.realpath(__file__)), os.pardir, "src"))
sys.path.append(libpath)

from data import QueryText, KeyWordSelectionDataset

file_name = path.expandvars("$DATASET_HOME/as_projet/quora.csv")
save_name_dc = path.expandvars("$DATASET_HOME/as_projet/quora_dataclasses.pkl")
save_name_torch = path.expandvars("$DATASET_HOME/as_projet/quora.torchdataset")

df = pd.read_csv(file_name)

english_stopwords = set(stopwords.words("english"))

punctu = re.compile(r'[^\w\s]')

dataclasses = []
i = 0
print("Computing QT")
for index, row in df.iterrows():
    print(index)
    q1 = row.question1
    q2 = row.question2
    
    q1 = q1.lower()

    q2 = q2.lower()
    q1 = expand_contractions(q1)
    q2 = expand_contractions(q2)

    q1 = punctu.sub(" ", q1)
    q2 = punctu.sub(" ", q2)

    q1 = [word for word in q1.split(" ") if len(word) > 0]
    q2 = [word for word in q2.split(" ") if len(word) > 0]

    q1_setclean = set(q1) - english_stopwords
    q2_setclean = set(q2) - english_stopwords

    common_words = q1_setclean & q2_setclean

    q1_dc = QueryText(str(i), common_words, q1)
    i += 1
    q2_dc = QueryText(str(i), common_words, q2)
    i += 1

    dataclasses.extend([q1_dc, q2_dc])

del df
del english_stopwords
gc.collect()

print("Model Building")
model = fastText.load_model("/local/pouyet/py37/models/wiki.en.bin")

print("embeddings")
for x in dataclasses:
    x.compute_embedding(model)

print("pickeling")
with open(save_name_dc, "wb") as f:
    pkl.dump(dataclasses, f)

dataset_torch = KeyWordSelectionDataset(querytext_list=dataclasses)
del dataclasses
gc.collect()
torch.save(dataset_torch, save_name_torch)

import sys
import os
from os import path

libpath = path.normpath(path.join(path.dirname(path.realpath(__file__)), os.pardir, "src"))
sys.path.append(libpath)

import elasticsearch as es
import pytrec_eval

from datasets import Robust2004
import numpy as np
import random

import torch
from torch.utils.data import DataLoader, random_split

def msearch_preprocess(query_texts, index="robust2004", doc_type="trec"):
    body = []

    header = {"index": index, "type": doc_type}
    for query_txt in query_texts:
        body.append(header)
        # query text needs to be a string
        query = {"size": MAX_DOC, "query": {"query_string": {"query": " ".join(query_txt), "default_field": "text"}}}
        body.append(query)
    
    return  body


def retrieve_doc_ids(hits):
    ret = {hit["_id"]: hit["_score"] for hit in hits}
    return ret


MAX_DOC = 1000
index = "robust2004-0.5-1"
doc_type = "trec"

seed = 5652
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


train_len = 200
dataset = Robust2004.torch_dataset()
dataclasses = Robust2004.dataclasses()
dataclasses = {dc._id: dc for dc in dataclasses}

trainset, testset = random_split(dataset, [train_len, len(dataset) - train_len])
train_ids = [str(r[2].long().tolist()) for r in trainset]
test_ids = [str(r[2].long().tolist()) for r in testset]
print(len(dataclasses))

train_queries = [v.query for k, v in dataclasses.items() if k in train_ids] 
test_queries = [v.query for k, v in dataclasses.items() if k in test_ids] 

train_qrel = {str(id_): dataclasses[id_].qrels for id_ in train_ids}
test_qrel = {str(id_): dataclasses[id_].qrels for id_ in test_ids}

train_msearch_body = msearch_preprocess(train_queries, index, doc_type)
test_msearch_body = msearch_preprocess(test_queries, index, doc_type)
print("search")
print(test_msearch_body)
engine = es.Elasticsearch()
train_res = engine.msearch(train_msearch_body, index)["responses"]
test_res = engine.msearch(test_msearch_body, index)["responses"]

print("id retrieval")
train_doc_ids = []
for resp in train_res:
    try:
        train_doc_ids.append(retrieve_doc_ids(resp["hits"]["hits"]))
    except:
        print("error")
        train_doc_ids.append({})
print("test")
test_doc_ids = []
for resp in test_res:
    try:
        test_doc_ids.append(retrieve_doc_ids(resp["hits"]["hits"]))
    except:
        print("error")
        test_doc_ids.append({})


#doc_ids = [retrieve_doc_ids(resp["hits"]["hits"]) for resp in res]
train_res_dict = dict(zip(train_ids, train_doc_ids))
test_res_dict = dict(zip(test_ids, test_doc_ids))

print("train eval")
evaluator = pytrec_eval.RelevanceEvaluator(train_qrel, set(("map",)))
# below line is the bottleneck
train_map_scores = evaluator.evaluate(train_res_dict)
train_map_scores = [a["map"] for a in train_map_scores.values()]
train_map_score = sum(train_map_scores) / len(train_map_scores)


print("test eval")
evaluator = pytrec_eval.RelevanceEvaluator(test_qrel, set(("map",)))
# below line is the bottleneck
test_map_scores = evaluator.evaluate(test_res_dict)
test_map_scores = [a["map"] for a in test_map_scores.values()]
test_map_score = sum(test_map_scores) / len(test_map_scores)

print("Train MAP :", train_map_score, "\tTest MAP :", test_map_score)

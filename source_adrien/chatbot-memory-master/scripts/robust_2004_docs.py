import sys
import os
from os import path

libpath = path.normpath(path.join(path.dirname(path.realpath(__file__)), os.pardir, "src"))
sys.path.append(libpath)

import elasticsearch as es
import pytrec_eval

from datasets import Robust2004


def msearch_preprocess(query_texts, index="robust2004", doc_type="trec"):
    body = []

    header = {"index": index, "type": doc_type}
    for query_txt in query_texts:
        body.append(header)
        # query text needs to be a string
        query = {"size": MAX_DOC, "query": {"query_string": {"query": " ".join(query_txt), "default_field": "text"}}}
        body.append(query)
    
    return  body


def get_results(qid, hits):
    res = []
    for rank, hit in enumerate(hits, 1):
        docid = hit["_id"]
        score = hit["_score"]
        res.append(" ".join(map(str, [qid, "Q0", docid, rank, score, "EARIA"])))
    return res


def retrieve_doc_ids(hits):
    ret = {hit["_id"]: hit["_score"] for hit in hits}
    return ret


MAX_DOC = 3000
index = "robust2004"
doc_type = "trec"

engine = es.Elasticsearch()

dataclasses = Robust2004.dataclasses()
dataclasses = {qt._id: qt for qt in dataclasses}
print(len(dataclasses))

queries = {str(k): v.query for k, v in dataclasses.items()} 

query_ids, query_texts = zip(*queries.items())
query_ids = list(map(str, query_ids))
qrel = {id_: dataclasses[id_].qrels for id_ in query_ids}
qrel = {str(k): v for k, v in qrel.items()}

msearch_body = msearch_preprocess(query_texts, index, doc_type)

res = [] 
for i in range(8): 
    res.extend(engine.msearch(msearch_body[i*50:i*50+50], index)["responses"]) 
res.extend(engine.msearch(msearch_body[400:], index)["responses"])

results = []
for qid, resp in zip(query_ids, res):
    results.extend(get_results(qid, resp["hits"]["hits"]))

with open("results.txt", "w") as f:
    f.write("\n".join(results))

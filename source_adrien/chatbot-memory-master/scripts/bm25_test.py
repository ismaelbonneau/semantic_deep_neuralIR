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


def retrieve_doc_ids(hits):
    ret = {hit["_id"]: hit["_score"] for hit in hits}
    return ret


MAX_DOC = 1000
index = "robust2004-0.5-1"
doc_type = "trec"

engine = es.Elasticsearch()

dataclasses = Robust2004.dataclasses()
dataclasses = {qt._id: qt for qt in dataclasses}
print(len(dataclasses))

queries = {str(k): v.get_text(v.rel) for k, v in dataclasses.items()}

query_ids, query_texts = zip(*queries.items())
query_ids = list(map(str, query_ids))
qrel = {id_: dataclasses[id_].qrels for id_ in query_ids}
qrel = {str(k): v for k, v in qrel.items()}

msearch_body = msearch_preprocess(query_texts, index, doc_type)

res = engine.msearch(msearch_body, index)["responses"]

doc_ids = []
for resp in res:
    try:
        doc_ids.append(retrieve_doc_ids(resp["hits"]["hits"]))
    except:
        print(resp)
        doc_ids.append({})



#doc_ids = [retrieve_doc_ids(resp["hits"]["hits"]) for resp in res]
res_dict = dict(zip(query_ids, doc_ids))

evaluator = pytrec_eval.RelevanceEvaluator(qrel, set(("map",)))
# below line is the bottleneck
map_scores = evaluator.evaluate(res_dict)
map_scores = [a["map"] for a in map_scores.values()]
map_score = sum(map_scores) / len(map_scores)
print(map_score)

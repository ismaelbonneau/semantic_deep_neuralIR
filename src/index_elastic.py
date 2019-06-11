"""
    auteur : Adrien Pouyet.
"""


import os, re
import subprocess
from itertools import accumulate
from typing import List, Dict, Tuple
import elasticsearch as es
import json
import numpy as np
import pytrec_eval



# Typings
QueryBody = Dict
Query = List[str]  # One word per item
Id = str
Score = float
Rel = int
Qrel = Tuple[Id, Rel]
Engine = es.Elasticsearch


MAX_DOC = 1000
default_index = "robust2004-0.5-1"
#  engine = get_engine(hosts=["big18:9200"])

def get_engine(**kwargs) -> Engine:
    """Get the engine.

    Parameters
    -----------
        kwargs : Parameters for the engine instantiation
    """
    engine = es.Elasticsearch(**kwargs)
    return engine


def _msearch_preprocess(query_texts: List[Query], index: str = default_index, doc_type: str = "trec") -> List[QueryBody]:
    """Preprocess the queries for the multi search with ElasticSearch

    Parameters
    ----------
        query_texts : the queries

        index : the index the search is done on

        doc_type : which document type is searched
    """
    body = []

    header = {"index": index, "type": doc_type}
    for query_txt in query_texts:
        body.append(header)
        query = {"size": MAX_DOC, "stored_fields": [],"query": {"query_string": {"query": " ".join(query_txt), "default_field": "text"}}}
        body.append(query)
    
    return body


def _retrieve_doc_ids(hits: dict) -> Dict[Id, Score]:
    ret = {hit["_id"]: hit["_score"] for hit in hits}
    return ret


def get_results(qid, hits):
    res = []
    for rank, hit in enumerate(hits, 1):
        docid = hit["_id"]
        score = hit["_score"]
        res.append(" ".join(map(str, [qid, "Q0", docid, rank, score, "EARIA"])))
    return res


def write_response(qids, response, resp_file):
    results = [f"{qid} Q0 EMPTY 1001 -100000 EARIA" for qid in qids]
    for qid, resp in zip(qids, response):
        results.extend(get_results(qid, resp["hits"]["hits"]))
    
    resp_file.write("\n".join(results))


def compute_trec_eval(qids, qrel_file_path, response, resp_file_name):
    with open(resp_file_name, "w") as f:
        write_response(qids, response, f)
    
    command = ["/local/karmim/logiciels/trec_eval/trec_eval", "-c", "-q", "-M1001", "-m", "map", qrel_file_path, resp_file_name]
    #Le fichier trec_eval est à récuperer ici https://github.com/usnistgov/trec_eval
    # une fois clone, il faut faire un make dans le dossier et autoriser à l'execution ( chmod +x)
    completed_process = subprocess.run(command, capture_output=True)
    results = completed_process.stdout.decode("utf-8")

    scores = {}
    for line in results.split("\n")[:-1]:
        temp = line.replace(" ", "").replace("\t", ",")
        _, qid, score = temp.split(",")
        scores[qid] = float(score)
    try:
        total_score = scores["all"]
        del scores["all"]
    except:
        total_score = 0
    
    return total_score, scores


def eval_queries(queries: Dict[Id, Query], qrel_file_path: str, engine: Engine,
        resp_file_path: str, index: str = default_index, doc_type: str = "trec") -> Score:
    """Eval the given queries according to the qrels using the engine."""
    query_ids, query_texts = zip(*queries.items())
    query_ids = list(map(str, query_ids))
    msearch_body = _msearch_preprocess(query_texts, index, doc_type)
    res = engine.msearch(msearch_body, index)
    res = res["responses"]
    print('ok')
    total_score, scores = compute_trec_eval(query_ids, qrel_file_path, res, resp_file_path)
    
    for qid in query_ids:
        if qid not in scores:
            scores[qid] = 0
    return total_score, scores

import ast

if __name__ == "__main__":
    engine = es.Elasticsearch(["http://big18:9200/"])
    print("engine ok :",engine)
    f = open("/local/karmim/Stage_M1_RI/data/robust2004.txt","r")
    #print(f.read())
    dico = ast.literal_eval(f.read())
    f.close()
    for k in dico:
        dico[k]= dico[k][0].split(' ') # On suppr les query langage naturel, et on met la query mot clé sous forme de liste.
    
    tot_score,scores = eval_queries(dico, qrel_file_path="/local/karmim/Stage_M1_RI/data/qrels.robust2004.txt", engine=engine,resp_file_path="/local/karmim/Stage_M1_RI/results/results.txt", index= "robust2004", doc_type= "trec")
    print("ecriture résultats ok")
    print("score MAP :" ,tot_score)
##########################################################

# Ismael Bonneau

# Creates an elasticSearch index 
# using BM25 with default parameters as a similarity measure

##########################################################


import elasticsearch as es
import elasticsearch.helpers as helpers

import os
from bs4 import BeautifulSoup
from parser_text import get_docno, get_text, get_title


def bulking(docs, index_name):
    bulk_data = [] 
    for doc in docs:
        data_dict = {
                '_index': index_name,
                '_id': get_docno(doc),
                '_source': {
                    "text": get_text(doc),
                    "title": get_title(doc)
                }
            }
        bulk_data.append(data_dict)
    return bulk_data # returns a list of python dicts representing documents

def collect_all_data(datasetpath, collections, index_name='robust-2004'):

    listedocs = []
    for collection in collections:
        for root, dirs, files in os.walk(datasetpath+os.sep+collection, topdown=True): # recursive walk on directories
            for name in files:
                with open(os.path.join(root, name), "r") as f:
                    filecontent = f.read()
                    soup = BeautifulSoup(filecontent, "html.parser")
                    docs = soup.find_all("doc") # get all "doc" tags and their content

                    yield bulking(docs, index_name) # memory-friendly iterator

#########################
        # script 
#########################

datasetpath = "/home/ismael/Documents/semantic_deep_neuralIR/data/collection"
collections = ["FR94", "FT", "FBIS", "LATIMES"]

# needs elasticSearch to be running
engine = es.Elasticsearch(["localhost:9200"])
assert engine.ping()

indexname = 'robust-2004'

# default parameters for BM25
b = 0.5
k1 = 1
settings = {"settings": {
                "number_of_shards": 1,
                "index": {
                    "similarity": {
                        "default": {
                            "type":"BM25",
                            "b": b,
                            "k1": k1}
                        }
                    },},
                "mappings": {         
                        "properties": {
                            "title": {
                                "type": "text",
                                "index": "false" # Do not index the title
                            },
                            "text": {
                                "type": "text",
                                "analyzer": "english"
                                }
                            }
                    }
                }

if indexname in engine.indices.get_alias("*"):
    engine.indices.delete(index=indexname)

engine.indices.create(indexname, settings)

for bulk_data in collect_all_data(datasetpath, collections, index_name=indexname):
	helpers.bulk(engine, bulk_data)

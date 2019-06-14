##########################################################
#
# script de création de l'index robust 2004
#
# Ismael Bonneau
##########################################################


import elasticsearch as es
import elasticsearch.helpers as helpers
from load_data import collect_all_data, bulking


datasetpath = "data/collection"
collections = ["FR94", "FT", "FBIS", "LATIMES"]

engine = es.Elasticsearch(["localhost:9200"])
assert engine.ping()

indexname = 'robust-2004'

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
                                "index": "false"
                            },
                            "text": {
                                "type": "text",
                                "analyzer": "english"
                                }
                            }
                    }
                }

engine.indices.delete(index='robust-2004')

#engine.indices.create(indexname, settings)

#for bulk_data in collect_all_data(datasetpath, collections, index_name=indexname):
	#helpers.bulk(engine, bulk_data)

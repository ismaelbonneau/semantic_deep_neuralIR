###############################################
#			Ismael Bonneau

# Retrieve 2000 top-ranked docs for every robust2004 query using elasticSearch's BM25
# for further use in LTR re-ranking

###############################################

import elasticsearch as es
import ast
import pickle

qrel_file_path = "/home/ismael/Documents/semantic_deep_neuralIR/data/qrels.robust2004.txt"
indexname = "robust-2004"

# elasticsearch connexion
engine = es.Elasticsearch([{'host':'localhost', 'port':9200}])
assert engine.ping()

def search(query, limit=2000):
	return engine.search(index=indexname, body={"_source": False,
	'size': limit,
	'query': {"query_string": {"query": query}}}, request_timeout=30)['hits']['hits']


with open(qrel_file_path, "r") as f:
	d_query = ast.literal_eval(f.read())

for k in d_query :
	d_query[k] = d_query[k][0] # getting rid of natural lang. queries, keeping only keywords


results_bm25 = {}

for id_requete, query in d_query.items():
	results = search(query)
	results_bm25[id_requete] = [doc["_id"] for doc in results]
	print("query {}.. done.".format(id_requete))

# saving results in a dictionnary like query_id -> list[doc_id]
pickle.dump(results_bm25, open("data/results_bm25_robust.pkl", 'wb'))  

import subprocess

# Check MAP and P@20 on this elasticSearch ranking:
resp_file_name = "tmp_bm25robust04"

results = [f"{id_requete} Q0 EMPTY 1001 -100000 EARIA" for id_requete in results_bm25]
for id_requete in results_bm25:
      res = []
      for i, (doc_id, score) in enumerate(results_bm25[:1000]):
            res.append(" ".join([id_requete, "Q0", doc_id, str(i + 1), str(score), "EARIA"]))
      results.extend(res)
with open(res_file, "w") as tiacompris:
      tiacompris.write("\n".join(results))


command = ["/home/ismael/Documents/semantic_deep_neuralIR/trec_eval" , "-c", "-M1001", "-m", "map","-m", "P.20", qrel_file_path, resp_file_name]

completed_process = subprocess.run(command, capture_output=True)
results = completed_process.stdout.decode("utf-8")
    
total_score = {}
for tkt in results.split("\n")[:2]:
      total_score[tkt.split("\t")[0].strip()] = float(tkt.split("\t")[2].strip())

print("elasticSearch MAP: %f P@20: %f" % (total_score["map"], total_score["P_20"]))                                                     
                                                                                                                                                                                                     
                                                                                                                                                                                                     

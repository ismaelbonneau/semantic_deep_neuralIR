import numpy as np
import pickle
import subprocess
import os
import torch

class Reranker:
    def __init__(self, bm25_dict):
        self.bm25_dict = bm25_dict #le dictionnaire query -> 2000 docs relevants pour bm25
        
    def set_model(self, model):
        self.model = model
    
    def rerank(self, queries=None):
        
        #queries: une liste d'ID de requete pour lesquels on veut réordonner les résultats
        if queries == None:
            queries_to_rerank = list(self.bm25_dict.keys())
        else:
            queries_to_rerank = queries
        
        #pour chaque requete, on va charger réordonner ses résultats avec le modèle
        query_idf = pickle.load(open("saved_data/query_idf.pkl", "rb"))
        reranked_dict = {}
        for id_requete in queries_to_rerank:
            if os.path.isfile("data/bm25_robust/{}_interractions.npy".format(id_requete)):
                #contient une matrice (2000, query_max_len, hist_size)
                saintjeanlapuenta = np.load("data/bm25_robust/{}_interractions.npy".format(id_requete))
                a = torch.from_numpy(np.tile(np.array([query_idf[id_requete]]), (saintjeanlapuenta.shape[0],1))).float()

                with torch.no_grad():
                    model_scores = self.model(torch.from_numpy(saintjeanlapuenta).float(), a).data.numpy()
                
                lol = np.argsort(model_scores)[::-1] #tri décroissant

                # reranked: liste de tuples (document_id, score)
                reranked_dict[id_requete] = [(self.bm25_dict[id_requete][i][0], model_scores[i]) for i in lol]

        return reranked_dict
    
    def get_results(self, id_requete, rank_list):
        results = []
        for i, (doc_id, score) in enumerate(rank_list[:1000]):
            results.append(" ".join([id_requete, "Q0", doc_id, str(i + 1), str(score), "EARIA"]))
        return results
            
    
    def save_results(self, rank_dict, res_file):
        """
        sauver sur un fichier au format attendu par TREC
        un dictionnaire query_id -> list (doc_id, score)
        """
        results = [f"{id_requete} Q0 EMPTY 1001 -100000 EARIA" for id_requete in rank_dict]
        for id_requete in rank_dict:
            results.extend(self.get_results(id_requete, rank_dict[id_requete]))
        
        with open(res_file, "w") as tiacompris:
            tiacompris.write("\n".join(results))


def compute_trec_eval(qrel_file_path, resp_file_name):
    command = ["trec_eval/trec_eval" , "-c", "-M1001", "-m", "map","-m", "P.20", qrel_file_path, resp_file_name]
    #Le fichier trec_eval est à récuperer ici https://github.com/usnistgov/trec_eval
    # une fois clone, il faut faire un make dans le dossier et autoriser à l'execution (chmod +x)
    completed_process = subprocess.run(command, capture_output=True)
    results = completed_process.stdout.decode("utf-8")
    
    total_score = {}
    for tkt in results.split("\n")[:2]:
        total_score[tkt.split("\t")[0].strip()] = float(tkt.split("\t")[2].strip())
    
    return total_score
import numpy as np
import matplotlib.pyplot as plt
from os import sep
import os
import random
import pickle

from functools import reduce
from reranking import Reranker, compute_trec_eval
import torch

seed = 1997
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


from drmm import * 

query_idf = pickle.load(open("saved_data/query_idf.pkl", "rb"))
folds = pickle.load(open("folds.pkl", 'rb'))

def get_train_test(i, folds):

    train_fold = folds.copy()
    test_fold = train_fold.pop(i)
    train_fold = reduce(lambda x,y :x+y, train_fold)

    train = {}
    test = {}

    for id_requete in test_fold:
        saintjeanlapuenta = np.load("saved_data/{}_interractions.npy".format(id_requete))
        interractions_test_pos = []
        interractions_test_neg = []
        for pos, neg in zip(saintjeanlapuenta[::2], saintjeanlapuenta[1::2]):
            interractions_test_pos.append([torch.from_numpy(pos).float(), torch.from_numpy(np.array([query_idf[id_requete]])).float()])
            interractions_test_neg.append([torch.from_numpy(neg).float(), torch.from_numpy(np.array([query_idf[id_requete]])).float()])
        test[id_requete]= {"pos": interractions_test_pos, "neg": interractions_test_neg}

    for id_requete in train_fold:
        interractions_train_pos = []
        interractions_train_neg = []
        saintjeanlapuenta = np.load("saved_data/{}_interractions.npy".format(id_requete))
        for pos, neg in zip(saintjeanlapuenta[::2], saintjeanlapuenta[1::2]):
            interractions_train_pos.append([torch.from_numpy(pos).float(), torch.from_numpy(np.array([query_idf[id_requete]])).float()])
            interractions_train_neg.append([torch.from_numpy(neg).float(), torch.from_numpy(np.array([query_idf[id_requete]])).float()])
        train[id_requete] = {"pos": interractions_train_pos, "neg": interractions_train_neg}

    #print("{} requetes en train, {} en test.".format(len(train), len(test)))
    return train, test


def get_random_sample_loader(train_dict, test_dict, batchsize=20):

    interractions_train_pos = []
    interractions_train_neg = []

    sample_size = 40
    for id_requete in train_dict:
        for i in np.random.choice(len(train_dict[id_requete]["pos"]), min(sample_size, len(train_dict[id_requete]["pos"])), replace=False):
            interractions_train_pos.append(train_dict[id_requete]["pos"][i])
        for i in np.random.choice(len(train_dict[id_requete]["neg"]), min(sample_size, len(train_dict[id_requete]["neg"])), replace=False):
            interractions_train_neg.append(train_dict[id_requete]["neg"][i])

    interractions_test_pos = []
    interractions_test_neg = []

    for id_requete in test_dict:
        for doc in test_dict[id_requete]["pos"]:
            interractions_test_pos.append(doc)
        for doc in test_dict[id_requete]["neg"]:
            interractions_test_neg.append(doc)

    train_dataset = DrmmDataset(interractions_train_pos, 
                                interractions_train_neg)
    
    val_dataset = DrmmDataset(interractions_test_pos,
                              interractions_test_neg)

    #classe utile pour gérer les mini batches et le shuffle (crucial!)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=len(interractions_test_neg), shuffle=False) 

    return train_loader, val_loader


def log(i, train_treceval, val_treceval, losses, val_losses):
    print("\tEPOCH {} ".format(i)+"train MAP: {0:.4f}".format(train_treceval['map'])+" val MAP : {0:.4f}".format(val_treceval['map'])+" val P@20: {0:.4f}".format(val_treceval['P_20'])+
        " *** "+"loss: {0:.4f}".format(losses[-1])+", val loss: {0:.4f}".format(val_losses[-1]))


qrel_file_path = "/home/ismael/Documents/semantic_deep_neuralIR/data/qrels.robust2004.txt"

loss_test = []
loss_train = []

maps_train = []
maps_test = []

bm25_dict = pickle.load(open("results_bm25_robust.pkl", "rb"))

for k in range(5):

    train_dict, test_dict = get_train_test(k, folds)

    drmm = DRMM(30, 4, hidden_sizes=[20,1])

    hingeloss = MarginRankingLoss(margin=1)
    optimizer = torch.optim.Adam(drmm.parameters(), lr=0.001)

    bm25_dict = pickle.load(open("results_bm25_robust.pkl", "rb"))
    reranker = Reranker(bm25_dict)
    reranker.set_model(drmm)

    make_train_step = drmm_make_train_step(drmm, hingeloss, optimizer)
    make_val_step = drmm_make_val_step(drmm, hingeloss, optimizer)

    losses = [] # loss train
    val_losses = [] # loss validation

    train_metrics = []
    val_metrics = [] # metriques de ri en validation

    nb_epochs = 100

    print("- fold {}".format(k))

    for i in range(nb_epochs):


        train_loader, val_loader = get_random_sample_loader(train_dict, test_dict, batchsize=20)
        #parcourir tous les batches du dataloader
        tiacompris = []
        for pos_batch, neg_batch in train_loader:

            #pos_batch[0]: les interractions positives
            #pos_batch[1]: les query term
            #neg_batch[0]: les interractions négatives correspondantes...
            #neg_batch[1]: les query term correspondantes...
            
            #évaluation empirical loss + backward pass
            tiacompris.append(make_train_step(pos_batch, neg_batch))

        losses.append(np.array(tiacompris).mean())
        #on met le modèle en mode "ne calcule pas le gradient"
        with torch.no_grad():
            #on parcourt
            tiacompris = []
            for pos_batch_val, neg_batch_val in val_loader:    
                tiacompris.append(make_val_step(pos_batch_val, neg_batch_val))
            val_losses.append(np.array(tiacompris).mean())

        if i % 10 == 0:
            reranked_dict_train = reranker.rerank()
            reranker.save_results(reranked_dict_train, "rerank_train")
            train_treceval = compute_trec_eval("qrels_train_fold{}.txt".format(k), "rerank_train")
            reranked_dict_test = reranker.rerank(list(test_dict.keys()))
            reranker.save_results(reranked_dict_test, "rerank_test")
            val_treceval = compute_trec_eval("qrels_test_fold{}.txt".format(k), "rerank_test")

            #log(i, train_treceval, val_treceval, losses, val_losses)

            train_metrics.append(train_treceval["map"])
            val_metrics.append(val_treceval["map"])


    maps_train.append(max(train_metrics))
    maps_test.append(max(val_metrics))
    loss_test.append(val_losses[-1])
    loss_train.append(losses[-1])

    print("\t map: " + str(max(val_metrics)))


print("train mean MAP: ", np.array(maps_train).mean(), " train std", np.array(maps_train).std())
print("val mean MAP: ", np.array(maps_test).mean(), " val std", np.array(maps_test).std())
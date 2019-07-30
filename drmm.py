import numpy as np
import torch
import torch.nn as nn
from torch.nn import MarginRankingLoss
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#hérite de la classe Pytorch Module
class DRMM(torch.nn.Module):
    def __init__(self, hist_size, query_term_maxlen, hidden_sizes=[5,1], use_cuda=True):
        super(DRMM, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(hist_size, hidden_sizes[0]), nn.Tanh(), 
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.Tanh())
        #initialisation du vecteur de term gating
        
        self.termgating = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        
    #méthode forward à redéfinir
    def forward(self, interractions, termvector):
        """
        interractions: (query_term_maxlen, hist_size)
        termvector: (1, query_term_maxlen)
        """
        #partie histogramme
        interractions_output = self.mlp(interractions).squeeze()
        # partie term gating
        gating_output = torch.nn.functional.softmax((self.termgating * termvector).squeeze(), dim=1)
        #combiner les 2 avec un produit scalaire
        axis = 1
        s = torch.sum(gating_output * interractions_output, dim = axis)
        return s
    
    def get_model_size(self):
        """
        retourne le nombre de paramètres du modèle.
        """
        return sum([p.size(0) if len(p.size()) == 1 else p.size(0)*p.size(1) for p in self.parameters()])
    
    def __str__(self):
        return "DRMM with {} parameters.".format(self.get_model_size())



class DrmmDataset(Dataset):
    """
    Cette classe va nous permettre de gérer
    le découpage en mini batches et le shuffle. 
    C'est un wrapper pour la classe DataSet 
    qui fournit entre autres un itérateur.
    """
    def __init__(self, pos_tensor, neg_tensor):
        self.x = pos_tensor
        self.y = neg_tensor
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    def __len__(self):
        return len(self.x)

def drmm_make_train_step(model, loss_fn, optimizer):
    def drmm_train_step(pos_batch, neg_batch):
        #mettre le modèle en mode train
        model.train()
        #étape forward...
        pos_score = model(pos_batch[0], pos_batch[1])
        neg_score = model(neg_batch[0], neg_batch[1])
        #calcul de la loss
        loss = loss_fn(pos_score, neg_score, torch.Tensor([1] * pos_batch[0].size()[0]))
        #calcul des gradients
        loss.backward()
        #mise à jour des paramètres
        optimizer.step()
        #reset des gradients après le passage sur ce batch
        optimizer.zero_grad()
        #retourner la loss
        return loss.item() #.item()
    
    return drmm_train_step

def drmm_make_val_step(model, loss_fn, optimizer):
    def drmm_val_step(pos_batch, neg_batch):
        #mettre le modèle en mode test
        model.eval()
        #étape forward...
        pos_score = model(pos_batch[0], pos_batch[1])
        neg_score = model(neg_batch[0], neg_batch[1])
        
        loss = loss_fn(pos_score, neg_score, torch.Tensor([1] * pos_batch[0].size()[0]))
        
        return loss.item()
    return drmm_val_step
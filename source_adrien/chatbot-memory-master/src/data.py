import os
from dataclasses import dataclass, field
from typing import List

from itertools import compress

import numpy as np
import torch
from nltk.stem import PorterStemmer
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader

from utils import filter_from

@dataclass
class QueryText:
    _id : str  # QueryId
    query : List  # List of words in the query
    text : List  # List of words of nlp query
    embeddings : np.ndarray = field(default_factory=lambda : np.array([]))
    rel : List = field(default_factory=list)
    qrels : List = field(default_factory=list)

    def __post_init__(self):
        self._compute_rel()
        
    def _compute_rel(self):
        if len(self.rel) == 0:
            ps = PorterStemmer()
            stemmed_querry = set(map(lambda x: ps.stem(x), self.query))
            self.rel = [int(ps.stem(word) in stemmed_querry) for word in self.text]
            print(self.query, stemmed_querry)
            print([int(word in self.query) for word in self.text], self.rel)
    
    def compute_embedding(self, model):
        embeddings = []
        for word in self.text:
            emb = model.get_word_vector(word)
            embeddings.append(emb)
        self.embeddings = np.array(embeddings)
    
    def get_text(self, selected_words):
        """Get the text of the selected words.

        Parameters
        -----------
            selected_words : list
                List of 0s and 1s whether the word is selected or not.
                len(selected_words) == len(self.text)
        
        Returns
        --------
            List of the selected words. The order is the same as in self.text
        """
        return list(compress(self.text, selected_words))


class KeyWordSelectionDataset(Dataset):
    def __init__(self, querytext_list=None):
        self.ids = torch.Tensor([int(qt._id) for qt in querytext_list])
        self.qrels = [qt.qrels for qt in querytext_list]

        embeddings = map(lambda x: torch.FloatTensor(x), (qt.embeddings for qt in querytext_list))
        self.embeddings = list(embeddings)
        rel = map(lambda x: torch.LongTensor(x), (qt.rel for qt in querytext_list))
        self.rel = list(rel)
    
    def __getitem__(self, i):
        return self.embeddings[i], self.rel[i], self.ids[i], self.qrels[i]
    
    def __len__(self):
        return len(self.ids)


def sequence_collate_fn(batch):
    x, y, ids, *_ = zip(*sorted(batch, key=lambda item: item[0].size(0), reverse=True))
    x = pad_sequence(x)
    y = pad_sequence(y)
    return x, y, ids


def embedding_collate_decorator(collate_fn):
    def wrapper(batch):
        x, y, ids = collate_fn(batch)
        return x, y, ids
    return wrapper

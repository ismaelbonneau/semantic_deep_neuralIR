import abc
import pickle as pkl
from os import path

import torch

from data import QueryText, KeyWordSelectionDataset

__all__ = ["Quora", "Robust2004"]

class Dataset(abc.ABC):
    @abc.abstractclassmethod
    def torch_dataset(cls):
        pass
    
    @abc.abstractclassmethod
    def dataclasses(cls):
        pass

class Quora(Dataset):
    dataclasses_path = path.expandvars("$DATASET_HOME/as_projet/quora_dataclasses.pkl")
    torch_path = path.expandvars("$DATASET_HOME/as_projet/quora.torchdataset")

    @classmethod
    def torch_dataset(cls):
        return torch.load(cls.torch_path)

    @classmethod
    def dataclasses(cls):
        with open(cls.dataclasses_path, 'rb') as f:
            dc = pkl.load(f)
        return dc


class Robust2004(Dataset):
    datasets_path = path.expandvars("$DATASET_HOME/as_projet/querries")
    dataclasses_path = path.join(datasets_path, "robust2004_dataclasses.pickled")
    torch_path = path.join(datasets_path, "robust2004.torchdataset")

    @classmethod
    def torch_dataset(cls):
        return torch.load(cls.torch_path)

    @classmethod
    def dataclasses(cls):
        with open(cls.dataclasses_path, 'rb') as f:
            dc = pkl.load(f)
        return dc
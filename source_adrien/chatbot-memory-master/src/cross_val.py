"""
Small module to do cross validation.
"""
import random
from itertools import chain, islice

from torch.utils.data import DataLoader, Subset

def k_fold(sequence, k=10):
    """Splits the sequence into 10 separate lists."""
    step = int(len(sequence) / k)
    folds = []
    sequence = iter(sequence)
    for _ in range(k-1):
        folds.append(list(islice(sequence, 0, step)))
    folds.append(list(islice(sequence, 0, None)))
    return folds


def all_but_one(sequence, k=10):
    """Select one vs all fold."""
    folds = k_fold(sequence, k)
    for i in range(k):
        prev = folds[0:i]
        actual = folds[i]
        then = folds[i+1:]
        yield chain(prev, then), actual

def cross_val(dataset, nb_fold, batch_size, collate_fn):
    """Given a dataset yields data for cross-validation.

    Parameters
    -----------
        dataset : torch.Dataset
            The dataset used

        nb_fold : int
            Number of fold used for cross-validation

        batch_size : int
            The batch size

    Yields
    ------
        trainloader : torch.Dataloader
        testloader : torch.Dataloader
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for trainindices, testindices in all_but_one(indices, k=nb_fold):
        trainindices = chain(*trainindices)
        trainset = Subset(dataset, list(trainindices))
        testset = Subset(dataset, list(testindices))
        trainloader = DataLoader(trainset, batch_size, shuffle=True, collate_fn=collate_fn)
        testloader = DataLoader(testset, batch_size, shuffle=True, collate_fn=collate_fn)

        yield trainloader, testloader

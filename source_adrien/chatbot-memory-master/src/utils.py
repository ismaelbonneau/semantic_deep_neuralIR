from itertools import chain, islice

import torch
import torch.nn as nn
from torch.distributions import Bernoulli


def fc_block(input_size, architecture, activation=nn.ReLU):
    """Build a fully connected block. Last layer has not activation function.

    Parameters
    ----------
        input_size : int
            Size of the input. Don't put it in the architecture

        architecture : list of int
            Each int of the list is the size of the corresponding layer

        activation_function : nn.Module
            Activation function after each layer of the net

    Returns
    ---------
        temp :  list of nn.Module
            The fully connected block without the activation function for the last layer
    """
    temp = []
    architecture = [input_size] + list(architecture)
    for prev_size, next_size in zip(architecture[:-1], architecture[1:]):
         temp.append(nn.Linear(prev_size, next_size))
         temp.append(nn.Dropout(.2)),
         temp.append(activation())
    temp = temp[:-1]  # Remove last activation
    return temp


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


class SepMemoryCell(nn.Module):
    """Recurrent cell for the SepMemory RNN.

    This cell has two cells :
        - one for positive (ie : 1) previous prediction
        - one for negative (ie : 0) previous prediction
    """
    def __init__(self, cell_archi, cell_type=nn.LSTM):
        super(SepMemoryCell, self).__init__()
        self.pos_cell = cell_type(**cell_archi)
        self.neg_cell = cell_type(**cell_archi)

    def forward(self, x, hidden, selected_prev_y):
        output_pos, hidden_pos = self.pos_cell(x, hidden)
        output_neg, hidden_neg = self.neg_cell(x, hidden)
        
        selected_prev_y = selected_prev_y.squeeze().unsqueeze(0).unsqueeze(-1).float()
        output = output_pos * selected_prev_y + output_neg * (1-selected_prev_y)
        
        new_hidden = []

        for h_pos, h_neg in zip(hidden_pos, hidden_neg):
            temp = h_pos * selected_prev_y + h_neg * (1-selected_prev_y)
            new_hidden.append(temp)

        return output.squeeze(), new_hidden


class SepMemoryRNN(nn.Module):
    """RNN using two seperate cells depending on previous output. Uses SepMemoryCell."""
    def __init__(self, predictor, decoder_archi):
        super(SepMemoryRNN, self).__init__()
        self.predictor = predictor
        self.cell = SepMemoryCell(decoder_archi)

    def forward(self, x, hidden=None, init_y=False):
        seq_len, batch_size = x.size(0), x.size(1)
        selected_prev_y = torch.LongTensor([init_y]*batch_size).to(x.device)

        dist_params = []
        actions = []
        hiddens = []
        for element in x:
            output, hidden = self.cell(element.unsqueeze(0), hidden, selected_prev_y)
            hiddens.append(hidden)
            pred = self.predictor(output)

            sampler = Bernoulli(pred)
            selected_prev_y = sampler.sample()
            
            actions.append(selected_prev_y)
            dist_params.append(pred)

        params = torch.stack(dist_params, dim=0).view(seq_len, batch_size)
        actions = torch.stack(actions, dim=0).view(seq_len, batch_size)
        return params, actions


class HierarchicalRNN(nn.Module):
    """Use a hierarchical RNN only if previous output was positive (ie : 1)."""
    def __init__(self, predictor, decoder_archi):
        super(HierarchicalRNN, self).__init__()
        self.decoder = nn.LSTM(**decoder_archi)
        self.hierarchical_decoder = nn.LSTM(**decoder_archi)
        self.predictor = predictor

        num_directions = 2 if decoder_archi["bidirectional"] else 1
        self.first_dim_hidden = decoder_archi["num_layers"]*num_directions
        self.hidden_size = decoder_archi["hidden_size"]

    def forward(self, x, hidden=None):
        seq_len, batch_size = x.size(0), x.size(1)
        if hidden is None:
            hidden = (torch.zeros(self.first_dim_hidden, batch_size, self.hidden_size).to(x.device),
                      torch.zeros(self.first_dim_hidden, batch_size, self.hidden_size).to(x.device))
        
        dist_params, actions = [], []
        for element in x:
            output, hidden_ = self.decoder(element.unsqueeze(0), hidden)
            pred = self.predictor(output)
            dist_params.append(pred)
            
            sampler = Bernoulli(pred)
            selected_prev_y = sampler.sample()
            
            actions.append(selected_prev_y)
            # bug ? below line was output, _ = ...
            output, hidden = self.hierarchical_decoder(element.unsqueeze(0), hidden)
            hidden = [h_ + h * pred for h_, h in zip(hidden_, hidden)]
        
        params = torch.stack(dist_params, dim=0).view(seq_len, batch_size)
        actions = torch.stack(actions, dim=0).view(seq_len, batch_size)
        return params, actions


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


def filter_from(function, selector, *iterables):
    """Given a function applied on the selector iterable,
       returns the items of each iterables where function(selector)
       is True

    Parameters
    ----------
        function : function
            The function applied on the selector iterator.
        
        selector : iterable
            The iterator which the function is applied on
        
        *iterables : list of iterable
            The iterables to select from
        
    Returns
    -------
        if function(selector[i]) == True:
            yield [it[i] for it in iterables]
    
    Example
    --------
        >>> selector = [0, 1, 1, 0]
        >>> args = [list(range(i, i+4)) for i in range(0, 12, 3)]  
        >>> print(args)
        -> [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9], [9, 10, 11, 12]]
        >>> gen = filter_from(lambda x: bool(x), selector, *args)
        >>> next(gen)
        -> (1, 4, 7, 10)
        >>> next(gen)
        -> (2, 5, 8, 11)
        >>> next(gen)
        -> StopIteration
    """
    for s, others in zip(selector, zip(*iterables)):
        if function(s):
            yield others

from itertools import starmap

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from utils import fc_block, Identity, SepMemoryCell, SepMemoryRNN, HierarchicalRNN


def apply_mask(tensor, mask):
    return tensor.squeeze() * mask

class KeyWordSelectionModel(nn.Module):
    """This is a base class for the following models.
    
    By inherating from this class, you have to implement the three methods :
        encode(x)
        decode(x)
        predict(x)
    
    The return of predict is what the forward returns so it should be a tensor of 0s and 1s
        0 if the word is not selected
        1 if the word is selected
    
    This design is made so it's easy to change the pipeline or the encoder, decoder and predictor
    without changing the entire code. That's what decorators at the end do.
    
    Attributes
    -----------
        encoder : nn.Module
            The module that encodes (thanks captain obvious)
        
        decoder : nn.Module
            The module that decodes what the encoder encoded
        
        predictor : nn.Module
            The module that predicts whether to select words or not
    """
    def __init__(self, encoder_archi=None, decoder_archi=None, predictor_structure=[]):
        """Instantiate the class.
        
        Parameters
        -----------
            encoder_archi : None or dict
                If dict :
                    Used to build an LSTM module so specify all obligatory parameters
                    with keywords. Please check torch.nn.LSTM documentation
                If None:
                    The encoder module is Idendity (returns the input)

            decoder_archi : None or dict
                If dict :
                    Used to build an LSTM module so specify all obligatory parameters
                    with keywords. Please check torch.nn.LSTM documentation
                If None:
                    The encoder module is Idendity (returns the input)
            
            predictor_structure :
                List of fully connected layers. Check utils.fc_block documentation.

        """
        super(KeyWordSelectionModel, self).__init__()
        self.encoder = Identity() if encoder_archi is None else nn.LSTM(**encoder_archi)
        self.decoder = Identity() if decoder_archi is None else nn.LSTM(**decoder_archi)
        self.predictor = nn.Sequential(*fc_block(predictor_structure[0], predictor_structure[1:]), nn.Sigmoid())

        self.encoder_archi = encoder_archi
        self.decoder_archi = decoder_archi
        self.predictor_structure = predictor_structure

        self.x_sizes = None

    def encode(self, x):
        raise NotImplementedError()

    def decode(self, x):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def forward(self, x, mask):
        self.x_sizes = x.size()
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        dist_params, actions = self.predict(decoded)
        self.x_sizes = None
        sampler = Bernoulli(dist_params)
        # Compute LogProba
        log_probas = sampler.log_prob(actions)
        log_probas = apply_mask(log_probas, mask)

        # Compute Entropy
        entropy = sampler.entropy()
        entropy = apply_mask(log_probas, mask)
        
        return actions, log_probas, entropy, dist_params


class KeyWordSelectionModel_1a(KeyWordSelectionModel):
    def __init__(self, decoder_archi, predictor_structure):
        super(KeyWordSelectionModel_1a, self).__init__(None, decoder_archi, predictor_structure)
        print("model 1a")

    def encode(self, x):
        return x

    def decode(self, x):
        output, (hidden, cell) = self.decoder(x)
        encoded = output.view(-1, output.size(-1))
        return encoded

    def predict(self, x):
        pred = self.predictor(x)
        pred = pred.view(self.x_sizes[0], self.x_sizes[1])
        return pred


class KeyWordSelectionModel_1b(KeyWordSelectionModel):
    def __init__(self, encoder_archi, decoder_archi, predictor_structure):
        super(KeyWordSelectionModel_1b, self).__init__(encoder_archi, decoder_archi, predictor_structure)
        print("model 1b")

    def encode(self, x):
        output, (hidden, cell) = self.encoder(x)
        return hidden, cell, x

    def decode(self, x):
        hidden, cell, x = x
        output, (hidden, cell) = self.decoder(x, [hidden, cell])
        encoded = output.view(-1, output.size(-1))
        return encoded

    def predict(self, x):
        pred = self.predictor(x)
        pred = pred.view(self.x_sizes[0], self.x_sizes[1])
        return pred


class KeyWordSelectionModel_1c(KeyWordSelectionModel):
    def __init__(self, encoder_archi, decoder_archi, predictor_structure):
        super(KeyWordSelectionModel_1c, self).__init__(encoder_archi, decoder_archi, predictor_structure)
        print("model 1c")

    def encode(self, x):
        output, (hidden, cell) = self.encoder(x)

        seq_len = self.x_sizes[0]
        batch = self.x_sizes[1]
        num_layers = self.encoder_archi["num_layers"]
        num_directions = 2 if self.encoder_archi["bidirectional"] else 1
        hidden_size = self.encoder_archi["hidden_size"]

        # Select last hidden and last cell to concatenate with x as the new input for the decoder
        last_hidden = hidden.view(num_layers, num_directions, batch, hidden_size)[0]
        last_hidden = last_hidden.view(batch, -1).unsqueeze(0).expand(seq_len, -1, -1)
        last_cell = cell.view(num_layers, num_directions, batch, hidden_size)[0]
        last_cell = last_cell.view(batch, -1).unsqueeze(0).expand(seq_len, -1, -1)

        new_x = torch.cat([x, last_hidden, last_cell], dim=-1)
        return new_x, [hidden, cell]

    def decode(self, x):
        new_x, (hidden, cell) = x
        output, (hidden, cell) = self.decoder(new_x, [hidden, cell])
        encoded = output.view(-1, output.size(-1))

        return encoded

    def predict(self, x):
        pred = self.predictor(x)
        pred = pred.view(self.x_sizes[0], self.x_sizes[1])
        return pred


###################
# Class decorator to add new mecanisms
###################

class IdentityRNN(nn.Module):
    def __init__(self):
        super(IdentityRNN, self).__init__()

    def forward(self, x, hiddens=(None, None)):
        return x, hiddens

class StochasticPredictor(nn.Module):
    def __init__(self, predictor):
        super(StochasticPredictor, self).__init__()
        self.predictor = predictor
        print("SP pred", predictor)
    
    def forward(self, *args, **kwargs):
        prob_params = self.predictor(*args, **kwargs)
        prob_params = prob_params.view(*prob_params.size()[:-1])
        sampler = Bernoulli(prob_params)
        actions = sampler.sample()
        return prob_params, actions


def memory_2a(model_class):
    """Memory 2a from the paper. Don't have the memory mecanism.
    
    Parameters
    -----------
        model_class : KeyWordSelectionModel
            The model to decorate
    """
    class Memory2a(model_class):
        def __init__(self, *args, **kwargs):
            super(Memory2a, self).__init__(*args, **kwargs)
            self.predictor = StochasticPredictor(self.predictor)
        
        def predict(self, x):
            print(x.size())
            x = x.view(self.x_sizes[0], self.x_sizes[1], -1)
            return self.predictor(x)
    print("memory 2a") 
    return Memory2a 


class Memory2b(KeyWordSelectionModel):
    def __init__(self, base_model):
        super(Memory2b, self).__init__()
        self.base_model = base_model
        self.base_model.predictor = SepMemoryRNN(self.predictor, self.decoder_archi)
        self.base_model.decoder = IdentityRNN()
    
    def encode(self, x):
        self.base_model.encode(x)
    
    def decode(self, x):
        self.base_model.decode(x)
    
    def predict(self, x):
        x = x.view(self.x_sizes[0], self.x_sizes[1], -1)
        res = self.predictor(x)
        return res 


def memory_2b(model_class):
    """Memory 2b from the paper.

    Use two separate decoder depending on whether previous word
    was selected or not.

    Parameters
    -----------
        model_class : KeyWordSelectionModel
            The model to decorate
    """
    class Memory2b(model_class):
        def __init__(self, *args, **kwargs):
            super(Memory2b, self).__init__(*args, **kwargs)
            self.predictor = SepMemoryRNN(self.predictor, self.decoder_archi)
            self.decoder = IdentityRNN()

        def predict(self, x):
            x = x.view(self.x_sizes[0], self.x_sizes[1], -1)
            res = self.predictor(x)
            return res 
    print("memory 2b")
    return Memory2b


def memory_2c(model_class):
    """Memory 2c from the paper.

    Use two hierarchical RNN if previous word was selected.

    Parameters
    -----------
        model_class : KeyWordSelectionModel
            The model to decorate
    """
    class Memory2C(model_class):
        """docstring for Memory2C."""
        def __init__(self, *args, **kwargs):
            super(Memory2C, self).__init__(*args, **kwargs)
            self.predictor = HierarchicalRNN(self.predictor, self.decoder_archi)
            self.decoder = IdentityRNN()

        def predict(self, x):
            x = x.view(self.x_sizes[0], self.x_sizes[1], -1)
            res = self.predictor(x)
            return res 
    print("memory 2c")
    return Memory2C

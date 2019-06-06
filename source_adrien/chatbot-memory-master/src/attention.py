import torch
import torch.nn as nn

from torch.distributions import Bernoulli

from transformer.Modules import ScaledDotProductAttention
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

from model import StochasticPredictor, apply_mask


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask
        
        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask
        
        return enc_output, enc_slf_attn


class TransformerEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_head, input_dim, d_inner, kq_size, v_size, n_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(input_dim, d_inner, n_head, kq_size, v_size, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, mask, return_attns=False):
        src_seq = src_seq.transpose(0, 1)
        mask = mask.transpose(0, 1)

        # -- Prepare masks
        slf_attn_mask = 1-mask.unsqueeze(1).expand(-1, src_seq.size(1), -1).byte()
        non_pad_mask = mask.unsqueeze(-1)

        # -- Forward
        enc_output = src_seq
        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        return enc_output


class AttModel(nn.Module):
    def __init__(self, n_head, d_inner, input_dim, kq_size, v_size, n_layers, dropout):
        super(AttModel, self).__init__()
        self.encoder = TransformerEncoder(n_head, d_inner, input_dim, kq_size, v_size, n_layers, dropout)
        
        predictor = nn.Sequential(nn.Linear(d_inner, 1), nn.Sigmoid())
        self.predictor = StochasticPredictor(predictor)

    def forward(self, seq, mask):
        encoded = self.encoder(seq, mask)
        dist_params, actions = self.predictor(encoded)
        dist_params, actions = dist_params.t(), actions.t()
        sampler = Bernoulli(dist_params)
        # Compute LogProba
        log_probas = sampler.log_prob(actions)
        log_probas = apply_mask(log_probas, mask)

        # Compute Entropy
        entropy = sampler.entropy()
        entropy = apply_mask(log_probas, mask)
        
        return actions, log_probas, entropy, dist_params


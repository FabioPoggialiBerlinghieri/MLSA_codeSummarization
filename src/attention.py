import numpy as np
import torch
import torch.nn as nn

class Attention(nn.Module):

    def __init__(self, d_model: int, input_dim: int = None):
        super().__init__()
        self.d_model = d_model
        self.input_dim = d_model if input_dim is None else input_dim
        self.W_Q = nn.Linear(self.input_dim, d_model)
        self.W_K = nn.Linear(self.input_dim, d_model)
        self.W_V = nn.Linear(self.input_dim, d_model)
        self.k = None
        self.v = None
        self.q = None
        self.scores = None

    def init_state(self, inputs: torch.Tensor):
        # input : B x L x D_input
        # dim k, v: B x L x D_Model
        self.k = self.W_K(inputs)
        self.v = self.W_V(inputs)

    def forward(self, query: torch.Tensor, mask: torch.Tensor =None):
        # q : B x L_q x D_Model
        self.q = self.W_Q(query)

        # score: B x L_q x D_Model * B x D_Model x L = B x L_q x L
        scores = torch.bmm(self.q, self.k.permute(0, 2, 1)) / np.sqrt(self.d_model)

        # mask : B x L_q x L
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # softmax for last dim
        self.scores = torch.softmax(scores, dim=-1)

        # Context: B x L_q x L * B x L x D_Model = B x L_q x D_Model
        return torch.bmm(self.scores, self.v)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def init_state(self, inputs: torch.Tensor):
        # input : B x L x Input
        assert inputs.dim() == 3
        assert inputs.shape[2] == self.input_dim

        #dim k, v: B x L x D_Model
        self.k = self.W_K(inputs)
        self.v = self.W_V(inputs)

    def forward(self, query: torch.Tensor, mask=None):
        # query : B x 1 x Input
        assert query.dim() == 3
        assert query.shape[1] == 1
        assert query.shape[2] == self.input_dim

        # q : B x 1 x D_Model
        self.q = self.W_Q(query)

        # Score B x L x D_Model * B x D_Model x 1 = B x L x 1
        scores = torch.bmm(self.k, self.q.permute(0, 2, 1)) / np.sqrt(self.d_model)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Context: B x 1 x L * B x L x D_Model = B x 1 x D_Model
        return torch.bmm(F.softmax(scores.permute(0, 2, 1)), self.v)

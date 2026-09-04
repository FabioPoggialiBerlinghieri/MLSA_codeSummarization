import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


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
        # input : B x L x Input
        if inputs.dim() != 3:
            raise ValueError(f"Expected a 3D tensor (B, L, Input), received a {inputs.dim()}D tensor.")
        if inputs.shape[2] != self.input_dim:
            raise ValueError(f"Incorrect input dimension: expected {self.input_dim}, received {inputs.shape[2]}.")

        #dim k, v: B x L x D_Model
        self.k = self.W_K(inputs)
        self.v = self.W_V(inputs)

    def forward(self, query: torch.Tensor, mask: torch.Tensor =None):
        self.__check_forward_params(query, mask)

        # q : B x L_q x D_Model
        self.q = self.W_Q(query)

        # score: B x L_q x D_Model * B x D_Model x L = B x L_q x L
        scores = torch.bmm(self.q, self.k.permute(0, 2, 1)) / np.sqrt(self.d_model)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # softmax for last dim
        self.scores = torch.softmax(scores, dim=-1)
        # Context: B x L_q x L * B x L x D_Model = B x L_q x D_Model
        return torch.bmm(self.scores, self.v)

    def __check_forward_params(self, query: Tensor, mask: Tensor | None):
        # state must be initialized
        if self.k is None or self.v is None:
            raise RuntimeError(
                "Attention state not initialized. You must call 'init_state(inputs)' before calling 'forward(query)'.")

        # query : B x L_q x Input
        if query.dim() != 3:
            raise ValueError(f"Expected query to be a 3D tensor (B, L_q, Input), but received a {query.dim()}D tensor.")
        if query.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected query feature dimension to match input_dim ({self.input_dim}), but received {query.shape[2]} at dimension 2.")

        # mask : B x L_q x L
        if mask is not None:
            if mask.dim() != 3:
                raise ValueError(f"Expected mask to be a 3D tensor (B, L_q, L), but received a {mask.dim()}D tensor.")
            if mask.shape[1] != query.shape[1]:
                raise ValueError(
                    f"Expected mask sequence length to be 1 (B, L_q, L), but received {mask.shape[1]} at dimension 1.")
            if mask.shape[2] != self.k.shape[1]:
                raise ValueError(
                    f"Mask sequence length must match input sequence length. Expected {self.k.shape[1]}, but received {mask.shape[2]} at dimension 2.")




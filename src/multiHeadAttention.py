import numpy as np
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int = 8, input_dim: int | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.input_dim = d_model if input_dim is None else input_dim

        self.W_Q = nn.Linear(self.input_dim, d_model)
        self.W_K = nn.Linear(self.input_dim, d_model)
        self.W_V = nn.Linear(self.input_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.k = None
        self.v = None
        self.q = None
        self.scores = None

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x : B x L x D_Model
        B, L, _ = x.size()

        # x : B x L x n_heads x head_dim
        x = x.view(B, L, self.n_heads, self.head_dim)

        # x : B x n_heads x L x head_dim
        x = x.transpose(1, 2)

        # x : (B * n_heads) x L x head_dim
        return x.reshape(B * self.n_heads, L, self.head_dim)

    def init_state(self, inputs: torch.Tensor) -> None:
        # input : B x L x D_input
        # k_full, v_full: B x L x D_Model
        k_full = self.W_K(inputs)
        v_full = self.W_V(inputs)

        # k, v : (B * n_heads) x L x head_dim
        self.k = self.split_heads(k_full)
        self.v = self.split_heads(v_full)

    def forward(self, query: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L_q, _ = query.size()

        # query : B x L_q x D_input
        # q_full : B x L_q x D_Model
        q_full = self.W_Q(query)

        # q : (B * n_heads) x L_q x head_dim
        self.q = self.split_heads(q_full)

        # score: (B * n_heads) x L_q x head_dim * (B * n_heads) x head_dim x L = (B * n_heads) x L_q x L
        scores = torch.bmm(self.q, self.k.permute(0, 2, 1)) / np.sqrt(self.head_dim)

        # mask : (B * n_heads) x L_q x L
        if mask is not None:
            mask = mask.repeat_interleave(self.n_heads, dim=0)
            scores = scores.masked_fill(mask == 0, -1e9)

        # softmax for last dim
        self.scores = torch.softmax(scores, dim=-1)

        # context: (B * n_heads) x L_q x L * (B * n_heads) x L x head_dim = (B * n_heads) x L_q x head_dim
        context = torch.bmm(self.scores, self.v)

        # context : B x n_heads x L_q x head_dim
        context = context.view(B, self.n_heads, L_q, self.head_dim)

        # context : B x L_q x n_heads x head_dim
        context = context.transpose(1, 2).contiguous()

        # context : B x L_q x D_Model
        context = context.view(B, L_q, self.d_model)

        # output : B x L_q x D_Model
        return self.out_proj(context)
import numpy as np
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, max_len : int, input_dim : int) -> None:
        super().__init__()
        self.input_dim = input_dim
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        angular_speed = torch.exp(torch.arange(0, input_dim, 2).float() * (-np.log(10000.0) / input_dim))
        pe[:, 0::2] = torch.sin(position * angular_speed)
        pe[:, 1::2] = torch.cos(position * angular_speed)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        scaled_x = x * np.sqrt(self.input_dim)
        encoded = scaled_x + self.pe[:, :x.size(1), :]
        return encoded
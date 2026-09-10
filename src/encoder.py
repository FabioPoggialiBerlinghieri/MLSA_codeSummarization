import torch.nn as nn
import torch
from attention import Attention

class SelfAttentionEncoder(nn.Module):

    def __init__(self, d_model: int, ff_dim: int | None = None) -> None:
        super().__init__()

        self.d_model = d_model
        self.ff_dim = d_model if ff_dim is None else ff_dim

        self.attention = Attention(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # should be a parameter
        dropout = 0.1
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, inputs_mask: torch.Tensor) -> torch.Tensor:

        # input: B x L_in x D_model
        # input_mask: B x L_in

        self.attention.init_state(inputs)

        # context: B x L_in x D_Model
        context = self.attention(inputs, inputs_mask)

        # dropout + skip conn
        context = self.norm1(self.dropout1(context) + inputs)

        # output:  B x L_in x D_Model
        outputs = self.ff(context)

        # dropout + skip conn
        return self.norm2(self.dropout2(outputs) + context)
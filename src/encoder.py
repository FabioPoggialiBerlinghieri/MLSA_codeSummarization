import torch.nn as nn
import torch
from attention import Attention
from paddingMask import PaddingMask

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

    def forward(self, inputs: torch.Tensor, inputs_mask: torch.Tensor) -> torch.Tensor:

        # input: B x L_in x D_model
        # input_mask: B x L_in

        self.attention.init_state(inputs)

        # context: B x L_in x D_Model
        context = self.attention(inputs, inputs_mask)

        # output:  B x L_in x D_Model
        return self.ff(context)
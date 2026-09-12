import warnings
import torch.nn as nn
import torch
from torch import Tensor
from decoder import Decoder
from encoder import SelfAttentionEncoder

class Transformer(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 1, n_heads: int = 1)-> None:
        super().__init__()
        self.encoders = nn.ModuleList([
            SelfAttentionEncoder(d_model, n_heads=n_heads) for _ in range(n_layers)
        ])
        self.decoders = nn.ModuleList([
            Decoder(d_model, n_heads=n_heads) for _ in range(n_layers)
        ])

    def forward(self, inputs: torch.Tensor,  inputs_mask: torch.Tensor,
                labels: torch.Tensor, labels_mask: torch.Tensor | None = None) -> torch.Tensor:

        # input : B x L_in x D_Model
        # input_mask: B x L_in
        # labels : B x L_label x D_Model
        # labels_mask: B x L_label

        # outputs encoder: B x L_in x D_Model
        enc_output = inputs
        for encoder in self.encoders:
            enc_output = encoder(enc_output, inputs_mask)

        for decoder in self.decoders:
            decoder.init_state(enc_output, inputs_mask)

        # output : B x L_label x D_Model
        dec_output = labels
        for decoder in self.decoders:
            dec_output = decoder(dec_output, labels_mask)

        return dec_output





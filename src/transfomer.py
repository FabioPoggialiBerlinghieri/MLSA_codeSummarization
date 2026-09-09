import warnings
import torch.nn as nn
import torch
from torch import Tensor
from decoder import Decoder
from encoder import SelfAttentionEncoder

class Transformer(nn.Module):
    def __init__(self, d_model: int)-> None:
        super().__init__()
        self.encoder = SelfAttentionEncoder(d_model)
        self.decoder = Decoder(d_model)

    def forward(self, inputs: torch.Tensor,  inputs_mask: torch.Tensor,
                labels: torch.Tensor, labels_mask: torch.Tensor | None = None) -> torch.Tensor:

        # input : B x L_in x D_Model
        # input_mask: B x L_in
        # labels : B x L_label x D_Model
        # labels_mask: B x L_label

        # outputs encoder: B x L_in x D_Model
        outputs_encoder = self.encoder(inputs, inputs_mask)

        self.decoder.init_state(outputs_encoder, inputs_mask)

        # output : B x L_label x D_Model
        outputs_decoder = self.decoder(labels,labels_mask)

        return outputs_decoder





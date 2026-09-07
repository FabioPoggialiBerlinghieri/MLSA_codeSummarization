import warnings
import torch.nn as nn
import torch
from torch import Tensor
from decoder import Decoder
from encoder import SelfAttentionEncoder

class Transformer(nn.Module):
    def __init__(self, d_model: int, feature_dim: int, output_max_len: int)-> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = SelfAttentionEncoder(d_model, input_dim=feature_dim)
        self.decoder = Decoder(d_model, input_dim=feature_dim, output_dim=feature_dim)
        self.output_max_len = output_max_len

    def forward(self, inputs: torch.Tensor,  inputs_mask: torch.Tensor,
                labels: torch.Tensor | None = None, labels_mask: torch.Tensor | None = None) -> torch.Tensor:
        # input : B x L_in x D_emb
        # labels : B x L_label x D_emb

        # outputs encoder: B x L_in x D_Model
        outputs_encoder = self.encoder(inputs, inputs_mask)
        # input mask must be of size: B x L_out x L_in
        inputs_mask = inputs_mask[:, 0:1, :].repeat(1, labels.shape[1], 1) if labels is not None else None
        self.decoder.init_state(outputs_encoder, inputs_mask)

        if self.training:
            # output : B x L_label x D_emb
            outputs_decoder = self.decoder(torch.cat([inputs[:, -1:, :], labels[:, :-1, :]], dim=1),
                                           labels_mask)
        else:
            current_sequence = inputs[:, -1:, :]

            for i in range(self.output_max_len):
                out = self.decoder(current_sequence)

                # B x 1 x D_emb
                next = out[:, -1:, :]
                current_sequence = torch.cat([current_sequence, next], dim=1)

            outputs_decoder = current_sequence[:, 1:, :]
            # output : B x L_label x D_emb

        return outputs_decoder





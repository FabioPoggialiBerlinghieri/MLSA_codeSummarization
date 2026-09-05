import warnings

import torch.nn as nn
import torch
from torch import Tensor

from decoder import Decoder
from encoder import SelfAttentionEncoder


class Transformer(nn.Module):
    def __init__(self, d_model: int, feature_dim: int, max_output_length: int)-> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = SelfAttentionEncoder(d_model, feature_dim)
        self.decoder = Decoder(d_model, feature_dim, feature_dim)
        self.max_output_length = max_output_length

    def forward(self, inputs: torch.Tensor,  inputs_seq_lengths: torch.Tensor,
                labels: torch.Tensor | None = None, labels_seq_lengths: torch.Tensor | None = None) -> torch.Tensor:
        self.__check_forward_params(inputs, labels)

        # outputs encoder: B x L x D_Model
        outputs_encoder = self.encoder(inputs, inputs_seq_lengths)
        self.decoder.init_state(outputs_encoder)

        if self.training:
            # output : B x L_label x Output
            outputs_decoder = self.decoder(torch.cat([inputs[:, -1:, :], labels[:, :-1, :]], dim=1),
                                           labels_seq_lengths)
        else:
            current_sequence = inputs[:, -1:, :]

            for i in range(self.max_output_length):
                out = self.decoder(current_sequence)
                # B x 1 x Feature
                next = out[:, -1:, :]
                current_sequence = torch.cat([current_sequence, next], dim=1)

            outputs_decoder = current_sequence[:, 1:, :]
            # output : B x L_max x Output

        return outputs_decoder

    def __check_forward_params(self, inputs: Tensor, labels: Tensor | None) -> None:
        # input : B x L x Input
        if inputs.dim() != 3:
            raise ValueError(f"Expected a 3D tensor (B, L, Input), received a {inputs.dim()}D tensor.")
        if inputs.shape[-1] != self.feature_dim:
            raise ValueError(f"Incorrect input dimension: expected {self.feature_dim}, received {inputs.shape[-1]}.")

        if not self.training and labels is not None:
            warnings.warn("Labels not request in eval mode")

        # labels : B x L_out x Output
        if labels is not None:
            if labels.dim() != 3:
                raise ValueError(f"Expected a 3D tensor (B, L, Input), received a {labels.dim()}D tensor.")
            if labels.shape[-1] != self.feature_dim:
                raise ValueError(
                    f"Incorrect labels dimension: expected {self.feature_dim}, received {labels.shape[-1]}.")

        if labels is None and self.training:
            raise ValueError("Labels must not be None in training mode")





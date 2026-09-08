from attention import Attention
import torch
import torch.nn as nn
from paddingMask import PaddingMask

class Decoder(nn.Module):
    def __init__(self, d_model: int, input_dim: int, output_dim: int, ff_dim: int | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim

        self.ff_dim = d_model if ff_dim is None else ff_dim

        self.self_attention = Attention(d_model, input_dim)
        self.cross_attention = Attention(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim),
        )

        self.inputs_mask = None

    def init_state(self, state: torch.Tensor, inputs_mask: torch.Tensor) -> None:
        # state : B x L_in x D_model
        self.cross_attention.init_state(state)
        self.inputs_mask = inputs_mask

    def forward(self, labels: torch.Tensor, labels_mask: torch.Tensor | None = None) -> torch.Tensor:
        # labels : B x L_label x D_emb
        # mask : B x L_label x L_label

        self.self_attention.init_state(labels)

        # add no cheating mask to padding mask
        if labels_mask is not None:
            labels_mask = labels_mask * torch.tril(torch.ones(labels.size(0), labels.size(1), labels.size(1), device=labels.device))

        # context : B x L_label x D_model
        context = self.self_attention(labels, labels_mask)

        # cross attention needs input mask
        context = self.cross_attention(context, self.inputs_mask)

        # output : B x L_labels x D_emb
        return self.ff(context)
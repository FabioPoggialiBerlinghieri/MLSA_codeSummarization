from attention import Attention
import torch
import torch.nn as nn
from paddingMask import PaddingMask

class Decoder(nn.Module):
    def __init__(self, d_model: int, ff_dim: int | None = None) -> None:
        super().__init__()
        self.d_model = d_model

        self.ff_dim = d_model if ff_dim is None else ff_dim

        self.self_attention = Attention(d_model)
        self.cross_attention = Attention(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, d_model),
        )

        self.inputs_mask = None

    def init_state(self, state: torch.Tensor, inputs_mask: torch.Tensor) -> None:
        # state : B x L_in x D_model
        self.cross_attention.init_state(state)
        self.inputs_mask = inputs_mask

    def forward(self, labels: torch.Tensor, labels_mask: torch.Tensor | None = None) -> torch.Tensor:

        # labels : B x L_label x D_Model
        # mask : B x L_label

        self.self_attention.init_state(labels)

        # mask for no cheating: B x L_label x L_label
        tril_mask = torch.tril(torch.ones(labels.size(0), labels.size(1), labels.size(1), device=labels.device))

        # add no cheating mask to padding mask
        # label_mask: B x L_label x L_label
        if labels_mask is not None:
            labels_mask = labels_mask * tril_mask
        else:
            labels_mask = tril_mask

        # context : B x L_label x D_Model
        context = self.self_attention(labels, labels_mask)

        # cross attention needs input mask
        # context : B x L_label x D_Model
        context = self.cross_attention(context, self.inputs_mask)

        # output : B x L_label x D_Model
        return self.ff(context)
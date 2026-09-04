import torch.nn as nn
import torch
from attention import Attention

class SelfAttentionEncoder(nn.Module):
    # Encoder with self-attention

    def __init__(self, d_model: int, input_dim: int, ff_dim: int | None = None) -> None:
        super().__init__()

        self.d_model = d_model
        self.input_dim = input_dim
        self.ff_dim = d_model if ff_dim is None else ff_dim

        self.attention = Attention(d_model, input_dim)
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, d_model),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self.__check_forward_params(inputs, mask)

        self.attention.init_state(inputs)
        # context: B x L x D_Model
        context = self.attention(inputs, mask)
        return self.ff(context)

    def __check_forward_params(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        # input : B x L x Input
        if inputs.dim() != 3:
            raise ValueError(f"Expected a 3D tensor (B, L, Input), received a {inputs.dim()}D tensor.")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"Incorrect input dimension: expected {self.input_dim}, received {inputs.shape[-1]}.")

        # mask : B x L x L
        if mask is not None:
            if mask.dim() != 3:
                raise ValueError(
                    f"Expected mask to be a 3D tensor (B, L, L), but received a {mask.dim()}D tensor.")
            if mask.shape[1] != inputs.shape[1] or mask.shape[2] != inputs.shape[1]:
                raise ValueError(
                    f"Expected mask to have shape (B, {inputs.shape[1]}, {inputs.shape[1]}), but received {mask.shape}.")

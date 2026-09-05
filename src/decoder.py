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
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, output_dim),
        )

    def init_state(self, state: torch.Tensor) -> None:
        # state : B x L x D
        if state.dim() != 3:
            raise ValueError(f"Expected state to be of size {3}, but found {state.dim()}")
        if state.shape[-1] != self.d_model:
            raise ValueError(f"Incorrect state dimension: expected {self.d_model}, received {state.shape[-1]}.")

        self.cross_attention.init_state(state)

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor | None = None) -> torch.Tensor:
        # input : B x L x Input
        if inputs.dim() != 3:
            raise ValueError(f"Expected input to be of size 3, but found {inputs.dim()}")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Incorrect input dimension: expected {self.input_dim}, received {inputs.shape[-1]}.")

        # mask : B x L x L
        mask = self.__create_mask(inputs, seq_lengths) if self.training else None

        self.self_attention.init_state(inputs)
        # context : B x L x D
        context = self.self_attention(inputs, mask)

        # context : B x L x D
        context = self.cross_attention(context)

        # output : B x L x Output
        return self.ff(context)

    def __create_mask(self, inputs: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        no_cheating_mask = torch.tril(torch.ones(inputs.size(0), inputs.size(1), inputs.size(1)))
        paddingMask = PaddingMask.generate_mask(inputs.shape[0], seq_lengths, inputs.shape[-2])
        return no_cheating_mask * paddingMask
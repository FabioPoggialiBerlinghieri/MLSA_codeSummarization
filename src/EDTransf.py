from ast import Tuple

import torch
from torch import nn

from positionalEncoding import PositionalEncoding
from transfomer import Transformer


class EDTransf(nn.Module):

    def __init__(self, d_model: int, input_max_len: int, input_vocabulary_size: int, embedding_dim: int, output_max_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        self.input_max_len = input_max_len
        self.preprocess = nn.Sequential(
            nn.Embedding(input_vocabulary_size, self.embedding_dim),
            PositionalEncoding(self.input_max_len, self.embedding_dim)
        )
        self.transformer = Transformer(d_model, self.embedding_dim, output_max_len)
        self.linear = nn.Linear(self.embedding_dim, 1)

    def forward(self, inputs: torch.Tensor, labels : torch.Tensor | None = None) -> torch.Tensor:

        # input : B x L_i x 1
        if inputs.shape[-2] != self.input_max_len:
            raise ValueError("Invalid input length")

        # seq_length : B
        inputs_seq_lengths, labels_seq_lengths = self.__create_seq_lengths(inputs, labels)

        # preprocessed: B x L_i x Emb_d
        preprocessed_inputs = self.preprocess(inputs)
        preprocessed_labels = self.preprocess(labels) if labels is not None else None

        # output: B x L_o x Emb_d
        outputs = self.transformer(preprocessed_inputs, inputs_seq_lengths, preprocessed_labels, labels_seq_lengths)

        # B x L_o x 1
        return self.linear(outputs)

    def __create_seq_lengths(self, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (inputs > 0).sum(dim=1), (labels > 0).sum(dim=1)

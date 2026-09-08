import torch
from torch import nn
from positionalEncoding import PositionalEncoding
from transfomer import Transformer

class EDTransf(nn.Module):

    def __init__(self, embedding_dim: int, d_model: int, input_max_len: int, input_vocabulary_size: int, output_max_len: int, output_vocabulary_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        self.input_max_len = input_max_len
        self.preprocess_inputs = nn.Sequential(
            nn.Embedding(input_vocabulary_size, self.embedding_dim),
            PositionalEncoding(self.input_max_len, self.embedding_dim)
        )
        self.preprocess_labels = nn.Sequential(
            nn.Embedding(output_vocabulary_size, self.embedding_dim),
            PositionalEncoding(self.input_max_len, self.embedding_dim)
        )

        self.transformer = Transformer(d_model, self.embedding_dim, output_max_len)
        # for each element of output seq we have a probabilistic distribution for a vocabulary size classification
        self.linear = nn.Linear(self.embedding_dim, output_vocabulary_size)

    def forward(self, inputs : torch.Tensor, input_mask : torch.Tensor,
                labels : torch.Tensor | None = None, labels_mask : torch.Tensor | None = None ) -> torch.Tensor:

        # inputs: B x L_in x 1 (code token)
        # labels: B x L_label x 1 (summ token)

        # preprocessed: B x L_i x D_emb
        preprocessed_inputs = self.preprocess_inputs(inputs)
        # preprocessed: B x L_label x D_emb
        preprocessed_labels = self.preprocess_labels(labels) if labels is not None else None

        # output: B x L_label x D_emb
        outputs = self.transformer(preprocessed_inputs, input_mask, preprocessed_labels, labels_mask)

        # B x L_label x output_vocabulary_size
        outputs = self.linear(outputs)

        # cross entropy want this parameters order
        return outputs.permute(0, 2, 1)

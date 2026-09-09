import torch
from torch import nn
from positionalEncoding import PositionalEncoding
from transfomer import Transformer

class EDTransf(nn.Module):

    def __init__(self, embedding_dim: int, input_max_len: int, input_vocabulary_size: int, output_max_len: int, output_vocabulary_size: int) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.input_max_len = input_max_len
        self.output_max_len = output_max_len

        self.preprocess_inputs = nn.Sequential(
            nn.Embedding(input_vocabulary_size, self.embedding_dim),
            PositionalEncoding(self.input_max_len, self.embedding_dim)
        )

        self.preprocess_labels = nn.Sequential(
            nn.Embedding(output_vocabulary_size, self.embedding_dim),
            PositionalEncoding(self.output_max_len, self.embedding_dim)
        )

        self.transformer = Transformer(self.embedding_dim)

        # for each element of output seq we have a probabilistic distribution for a vocabulary size classification
        self.linear = nn.Linear(self.embedding_dim, output_vocabulary_size)


    def predict(self, inputs: torch.Tensor, input_mask: torch.Tensor, cls: int, sep: int) -> torch.Tensor:

        assert not self.training

        # inputs: L_in (code token)
        # inputs_mask: L_in

        # inputs: B (=1) x L_in
        # inputs_mask: B (=1) x L_in
        # current_seq: B (=1) x L_lab (=1)
        inputs = torch.tensor([inputs], dtype=torch.long, device=inputs.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long, device=inputs.device)
        current_seq = torch.tensor([[cls]], dtype=torch.long, device=inputs.device)

        # preprocessed_inputs: B x L_in x D_emb
        preprocessed_inputs = self.preprocess_inputs(inputs)

        for i in range(self.output_max_len):

            # current_seq_preprocessed: B x L_label x D_emb
            current_seq_preprocessed = self.preprocess_labels(current_seq)

            # outputs: B x L_label x D_emb
            outputs = self.transformer(preprocessed_inputs, input_mask, current_seq_preprocessed)

            # outputs: B x L_label x output_vocabulary_size
            outputs = self.linear(outputs)

            # outputs: B x L_label
            outputs = torch.argmax(outputs, dim=-1)

            # last element of the generated sequence
            outputs = outputs[:, -1:]

            # concatenate at current_seq, L_label++
            current_seq = torch.cat([current_seq, outputs], dim=-1)

            if outputs.item() == sep:
                break

        # generate_len (upperbound max_output_len)
        return current_seq.squeeze(0)

    def forward(self, inputs : torch.Tensor, input_mask : torch.Tensor,
                labels : torch.Tensor, labels_mask : torch.Tensor) -> torch.Tensor:

        # inputs: B x L_in (code token)
        # inputs_mask: B x L_in
        # labels: B x L_label (summ token)
        # labels_mask: B x L_label

        # preprocessed: B x L_in x D_emb
        preprocessed_inputs = self.preprocess_inputs(inputs)

        # preprocessed: B x L_label x D_emb
        preprocessed_labels = self.preprocess_labels(labels)

        # output: B x L_label x D_emb
        outputs = self.transformer(preprocessed_inputs, input_mask, preprocessed_labels, labels_mask)

        # B x L_label x output_vocabulary_size
        outputs = self.linear(outputs)

        # cross entropy want this parameters order
        return outputs.permute(0, 2, 1)

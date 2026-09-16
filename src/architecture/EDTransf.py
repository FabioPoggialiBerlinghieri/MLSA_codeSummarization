import torch
from torch import nn
from .positionalEncoding import PositionalEncoding
from .transformer import Transformer

class EDTransf(nn.Module):

    def __init__(self, embedding_dim: int, input_max_len: int, input_vocabulary_size: int, output_max_len: int, output_vocabulary_size: int,
                 n_layers: int = 1, n_heads: int = 1, dropout: float = 0.1) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.input_max_len = input_max_len
        self.output_max_len = output_max_len
        self.output_vocabulary_size = output_vocabulary_size

        self.preprocess_inputs = nn.Sequential(
            nn.Embedding(input_vocabulary_size, self.embedding_dim),
            PositionalEncoding(self.input_max_len, self.embedding_dim)
        )

        self.preprocess_labels = nn.Sequential(
            nn.Embedding(output_vocabulary_size, self.embedding_dim),
            PositionalEncoding(self.output_max_len, self.embedding_dim)
        )

        self.transformer = Transformer(self.embedding_dim, n_layers=n_layers, n_heads=n_heads, dropout=dropout)

        # linear layer mapping embeddings to vocabulary size for output probability distribution
        self.linear = nn.Linear(self.embedding_dim, output_vocabulary_size)

    def predict(self, inputs: torch.Tensor, input_mask: torch.Tensor, cls: int, sep: int, pad: int) -> torch.Tensor:
        """
        Generates summaries using greedy decoding.
        Selects the token with the highest probability at each generation step.
        """
        assert not self.training

        # inputs: B x L_in (code token)
        # inputs_mask: B x L_in

        batch_size = inputs.size(0)
        device = inputs.device

        # initialize sequence with CLS token
        current_seq = torch.full((batch_size, 1), cls, dtype=torch.long, device=device)

        # preprocessed_inputs: B x L_in x D_emb
        preprocessed_inputs = self.preprocess_inputs(inputs)

        enc_output = self.transformer.encode(preprocessed_inputs, input_mask)
        self.transformer.init_decoders(enc_output, input_mask)

        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(self.output_max_len):

            # current_seq_preprocessed: B x L_label x D_emb
            current_seq_preprocessed = self.preprocess_labels(current_seq)

            # outputs: B x L_label x D_emb
            outputs = self.transformer.decode(current_seq_preprocessed)

            # outputs: B x L_label x output_vocabulary_size
            outputs = self.linear(outputs)

            # new tokens generated (greedy selection)
            next_tokens = torch.argmax(outputs[:, -1:, :], dim=-1)

            # fill with PAD token if the sentence is already finished
            next_tokens = next_tokens.masked_fill(is_finished.unsqueeze(1), pad)

            # concatenate at current_seq, L_label++
            current_seq = torch.cat([current_seq, next_tokens], dim=-1)

            # update finished status if SEP token is generated
            just_finished = (next_tokens.squeeze(1) == sep)
            is_finished = is_finished | just_finished

            if is_finished.all():
                break

        # current_seq: generate_len (upperbound max_output_len)
        return current_seq

    def predict_bs(self, inputs: torch.Tensor, input_mask: torch.Tensor, cls: int, sep: int, pad: int,
                   beam_width: int = 3, alpha: float = 0.8) -> torch.Tensor:
        """
        Generates summaries using Beam Search decoding.
        Keeps track of multiple candidates (beam width) to find the most probable sequence.
        """
        assert not self.training

        # inputs: B x L_in (code token)
        # inputs_mask: B x L_in

        batch_size = inputs.size(0)
        device = inputs.device

        # preprocessed_inputs: B x L_in x D_emb
        preprocessed_inputs = self.preprocess_inputs(inputs)

        enc_output = self.transformer.encode(preprocessed_inputs, input_mask)

        # enc_output: B*W x L_in x D_emb
        enc_output = enc_output.repeat_interleave(beam_width, dim=0)
        # inputs_mask: B*W x L_in
        input_mask = input_mask.repeat_interleave(beam_width, dim=0)

        self.transformer.init_decoders(enc_output, input_mask)

        # Initialize sequence with CLS token
        # current_seq: B*W x 1
        current_seq = torch.full((batch_size * beam_width, 1), cls, dtype=torch.long, device=device)

        # track finished sequences
        # is_finished: B*W
        is_finished = torch.zeros(batch_size * beam_width, dtype=torch.bool, device=device)

        # track sequence lengths for penalty calculation
        # seq_lengths: B*W
        seq_lengths = torch.ones(batch_size * beam_width, dtype=torch.long, device=device)

        # beam_scores: B*W
        beam_scores = torch.full((batch_size, beam_width), float('-inf'), device=device)
        beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.view(-1)

        for i in range(self.output_max_len):

            # current_seq_preprocessed: B*W x L_label x D_emb
            current_seq_preprocessed = self.preprocess_labels(current_seq)

            # outputs: B*W x L_label x D_emb
            outputs = self.transformer.decode(current_seq_preprocessed)

            # outputs: B*W x L_label x output_vocabulary_size
            outputs = self.linear(outputs)

            # next_token_logits: B*W x vocab_size
            next_token_logits = outputs[:, -1, :]

            current_seq, beam_scores, is_finished, seq_lengths = self.beam_search(
                current_seq=current_seq,
                beam_scores=beam_scores,
                next_token_logits=next_token_logits,
                is_finished=is_finished,
                seq_lengths=seq_lengths,
                alpha=alpha,
                pad=pad, sep=sep,
                batch_size=batch_size,
                beam_width=beam_width
            )

            if is_finished.all():
                break

        # reshape to separate batch size and beam width
        # current_seq: B x W x L_label
        current_seq = current_seq.view(batch_size, beam_width, -1)

        # topk of beam_search returns ordered results, so the first seq is the best one
        # best_seq: B x L_label
        best_seq = current_seq[:, 0, :]

        return best_seq

    def beam_search(self, current_seq: torch.Tensor, beam_scores: torch.Tensor, next_token_logits: torch.Tensor,
                    is_finished: torch.Tensor,
                    seq_lengths: torch.Tensor, alpha: float, pad: int, sep: int, batch_size: int, beam_width: int):

        # current_seq: B*W

        vocab_size = next_token_logits.size(-1)
        device = current_seq.device

        next_token_log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

        # prevent finished sentences from selecting new tokens by setting probability to -inf
        next_token_log_probs[is_finished] = float('-inf')

        # allow finished sentences to generate PAD tokens without score penalty
        next_token_log_probs[is_finished, pad] = 0.0

        # add old score with the new one (raw scores)
        # raw_scores: B*W x voc_size
        raw_scores = beam_scores.unsqueeze(1) + next_token_log_probs

        # update lengths
        # new_lengths: B*W x voc_size
        new_lengths = seq_lengths.unsqueeze(1).repeat(1, vocab_size)
        new_lengths[~is_finished] += 1

        # normalized_scores: B*W x voc_size
        normalized_scores = raw_scores / (new_lengths ** alpha)

        # reshape for topk: B x W*voc_size
        normalized_scores = normalized_scores.view(batch_size, beam_width * vocab_size)
        raw_scores = raw_scores.view(batch_size, beam_width * vocab_size)
        new_lengths = new_lengths.view(batch_size, beam_width * vocab_size)

        # find best W path using normalized scores
        # top_scores: B x W
        # top_indices: B x W
        top_scores, top_indices = torch.topk(normalized_scores, k=beam_width, dim=1)

        # update beam_scores with raw scores to not break math
        # new_beam_scores: B*W
        new_beam_scores = torch.gather(raw_scores, 1, top_indices).view(-1)

        # update seq_lengths
        # new_seq_lengths: B*W
        new_seq_lengths = torch.gather(new_lengths, 1, top_indices).view(-1)

        # find tokens (position of vocab size)
        # next_tokens: W*B x 1
        next_tokens = (top_indices % vocab_size).view(-1, 1)
        # find beam of top indices
        # beam_indices: B x W
        beam_indices = top_indices // vocab_size

        # batch_indices: B x 1
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)

        # select indices of current_seq vector, that represent position of the path of the news tokens
        # global_beam_indices: B*W
        global_beam_indices = (batch_indices * beam_width + beam_indices).view(-1)

        new_current_seq = current_seq[global_beam_indices]
        new_is_finished = is_finished[global_beam_indices]

        # concat current_seq and new token
        new_current_seq = torch.cat([new_current_seq, next_tokens], dim=-1)

        # update finished
        just_finished = (next_tokens.squeeze(1) == sep)
        new_is_finished = new_is_finished | just_finished

        return new_current_seq, new_beam_scores, new_is_finished, new_seq_lengths

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

        # cross entropy wants this parameters order
        return outputs.permute(0, 2, 1)
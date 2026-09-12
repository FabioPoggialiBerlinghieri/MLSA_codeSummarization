import unittest
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from multiHeadAttention import MultiHeadAttention

class MultiHeadAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.attention = MultiHeadAttention(d_model=4, n_heads=2)
        self.attention.eval()

        self.W_Q = self.attention.W_Q.weight
        self.b_Q = self.attention.W_Q.bias
        self.W_K = self.attention.W_K.weight
        self.b_K = self.attention.W_K.bias
        self.W_V = self.attention.W_V.weight
        self.b_V = self.attention.W_V.bias
        self.W_O = self.attention.out_proj.weight
        self.b_O = self.attention.out_proj.bias

        # L_q = 1 nei test

    def test_init_state(self):
        # input : B x L x D_model = 1 x 3 x 4
        inputs = torch.tensor([[[1, 2, 1, -1], [2, -1, 0, 1], [-1, 3, 2, 2]]]).float()

        self.attention.init_state(inputs)

        # La GPU mergia Batch e n_heads: 1 (B) * 2 (n_heads) = 2.
        # La dimensione finale deve essere (B*n_heads, L, head_dim) -> (2, 3, 2)
        self.assertEqual((2, 3, 2), self.attention.k.shape)
        self.assertEqual((2, 3, 2), self.attention.v.shape)

        # Verifica logica dello split
        k_expected = F.linear(inputs, self.W_K, self.b_K)

        # La Testa 1 deve essersi presa esattamente la prima metà (i primi 2 elementi)
        torch.testing.assert_close(k_expected[0, :, :2], self.attention.k[0])
        # La Testa 2 deve essersi presa la seconda metà
        torch.testing.assert_close(k_expected[0, :, 2:], self.attention.k[1])

    def test_forward_q_value(self):
        inputs = torch.tensor([[[1, 2, 1, -1], [2, -1, 0, 1], [-1, 3, 2, 2]]]).float()
        self.attention.init_state(inputs)

        # query : B x 1 x D_model = 1 x 1 x 4
        query = torch.tensor([[[1, 2, -1, 1]]]).float()

        self.attention(query)

        # (B*n_heads) x L_q x head_dim
        self.assertEqual((2, 1, 2), self.attention.q.shape)

        q_expected = F.linear(query, self.W_Q, self.b_Q)
        torch.testing.assert_close(q_expected[0, :, :2], self.attention.q[0])

    def test_forward_scores_value(self):
        inputs = torch.tensor([[[1, 2, 1, -1], [2, -1, 0, 1], [-1, 3, 2, 2]]]).float()
        self.attention.init_state(inputs)

        query = torch.tensor([[[1, 2, -1, 1]]]).float()

        self.attention(query)

        # score: (B*n_heads) x L_q x L = 2 x 1 x 3
        self.assertEqual((2, 1, 3), self.attention.scores.shape)

        # Ricalcoliamo manualmente gli score solo per la Testa 1 per conferma
        q_head1 = self.attention.q[0:1]  # 1 x 1 x 2
        k_head1 = self.attention.k[0:1]  # 1 x 3 x 2
        expected_scores_head1 = torch.softmax(torch.bmm(q_head1, k_head1.permute(0, 2, 1)) / np.sqrt(2), dim=-1)

        torch.testing.assert_close(expected_scores_head1[0], self.attention.scores[0])

    def test_forward(self):
        nn.init.eye_(self.attention.W_Q.weight)
        nn.init.eye_(self.attention.W_K.weight)
        nn.init.eye_(self.attention.W_V.weight)
        nn.init.eye_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.W_Q.bias)
        nn.init.zeros_(self.attention.W_K.bias)
        nn.init.zeros_(self.attention.W_V.bias)
        nn.init.zeros_(self.attention.out_proj.bias)

        inputs = torch.tensor([[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]])
        self.attention.init_state(inputs)

        query = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])

        context = self.attention(query)

        # stessi riusltati test attention
        expected_context = torch.tensor([[[0.6697, 0.3303, 0.6697, 0.3303]]])

        self.assertEqual((1, 1, 4), context.shape)
        torch.testing.assert_close(context, expected_context, rtol=1e-3, atol=1e-3)

    def test_forward_scores_value_with_mask(self):
        inputs = torch.tensor([[[1, 2, 1, -1], [2, -1, 0, 1], [-1, 3, 2, 2]]]).float()
        self.attention.init_state(inputs)

        query = torch.tensor([[[1, 2, -1, 1]]]).float()

        # mask : B x 1 x L
        mask = torch.tensor([[[1, 0, 0]]]).float()

        self.attention(query, mask=mask)

        self.assertEqual((2, 1, 3), self.attention.scores.shape)

        q_head1 = self.attention.q[0:1]
        k_head1 = self.attention.k[0:1]
        a_head1 = torch.bmm(q_head1, k_head1.permute(0, 2, 1)) / np.sqrt(2)

        expected_scores_head1 = torch.softmax(torch.tensor([[[a_head1[:, :, 0].item(), -1e9, -1e9]]]).float(), dim=-1)

        torch.testing.assert_close(expected_scores_head1[0], self.attention.scores[0])


if __name__ == '__main__':
    unittest.main()
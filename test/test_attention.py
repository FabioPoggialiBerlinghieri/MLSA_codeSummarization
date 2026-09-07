import unittest
import torch
import torch.nn.functional as F
from attention import Attention
import torch.nn as nn
import numpy as np

class AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.attention = Attention(2, 2)
        self.attention.eval()

        self.W_Q = self.attention.W_Q.weight
        self.b_Q = self.attention.W_Q.bias
        self.W_K = self.attention.W_K.weight
        self.b_K = self.attention.W_K.bias
        self.W_V = self.attention.W_V.weight
        self.b_V = self.attention.W_V.bias

        # in this tests L_q = 1

    def test_init_state(self):
        # input : B x L x Input = 1 x 3 x 2
        inputs = torch.tensor([[[1,2],[2,-1],[-1,3]]]).float()

        self.attention.init_state(inputs)

        self.assertEqual((1, 3, 2), self.attention.k.shape)
        self.assertEqual((1, 3, 2), self.attention.v.shape)

        torch.testing.assert_close(F.linear(inputs, self.W_K, self.b_K), self.attention.k)
        torch.testing.assert_close(F.linear(inputs, self.W_V, self.b_V), self.attention.v)

    def test_forward_q_value(self):
        # input : B x L x Input = 1 x 3 x 2
        inputs = torch.tensor([[[1, 2], [2, -1], [-1, 3]]]).float()
        self.attention.init_state(inputs)

        # query : B x 1 x Input = 1 x 1 x 2
        query = torch.tensor([[[1,2]]]).float()

        self.attention(query)

        self.assertEqual((1, 1, 2), self.attention.q.shape)
        torch.testing.assert_close(F.linear(query, self.W_Q, self.b_Q), self.attention.q)

    def test_forward_scores_value(self):
        # input : B x L x Input = 1 x 3 x 2
        inputs = torch.tensor([[[1, 2], [2, -1], [-1, 3]]]).float()
        self.attention.init_state(inputs)

        # query : B x 1 x Input = 1 x 1 x 2
        query = torch.tensor([[[1,2]]]).float()

        self.attention(query)

        self.assertEqual((1, 1, 3), self.attention.scores.shape)

        q_expected = F.linear(query, self.W_Q, self.b_Q)
        k_expected = F.linear(inputs, self.W_K, self.b_K)

        expected_scores = torch.softmax(torch.bmm(q_expected, k_expected.permute(0, 2, 1)) / np.sqrt(self.attention.d_model), dim=-1)

        torch.testing.assert_close(expected_scores, self.attention.scores)

    def test_forward(self):
        # set eye as Q and zeros as bias
        nn.init.eye_(self.attention.W_Q.weight)
        nn.init.eye_(self.attention.W_K.weight)
        nn.init.eye_(self.attention.W_V.weight)
        nn.init.zeros_(self.attention.W_Q.bias)
        nn.init.zeros_(self.attention.W_K.bias)
        nn.init.zeros_(self.attention.W_V.bias)

        inputs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        self.attention.init_state(inputs)

        query = torch.tensor([[[1.0, 0.0]]])

        context = self.attention(query)

        # Q * K^T :
        # [1, 0] * [1, 0] = 1.0
        # [1, 0] * [0, 1] = 0.0
        #
        # scale for sqrt(d_model) -> sqrt(2) = 1.414
        # -> [1.0 / 1.414, 0.0] = [0.7071, 0.0]
        #
        # Softmax di [0.7071, 0.0]:
        # e^0.7071 / (e^0.7071 + e^0) = 2.028 / (2.028 + 1) = 0.6697
        # e^0      / (e^0.7071 + e^0) = 1 / 3.028 = 0.3303
        #
        # context:
        # 0.6697 * [1, 0] + 0.3303 * [0, 1] = [0.6697, 0.3303]

        expected_context = torch.tensor([[[0.6697, 0.3303]]])
        torch.testing.assert_close(context, expected_context, rtol=1e-3, atol=1e-3)

    def test_forward_scores_value_with_mask(self):
        # input : B x L x Input = 1 x 3 x 2
        inputs = torch.tensor([[[1, 2], [2, -1], [-1, 3]]]).float()
        self.attention.init_state(inputs)

        # query : B x 1 x Input = 1 x 1 x 2
        query = torch.tensor([[[1,2]]]).float()

        # mask : B x 1 x L
        mask = torch.tensor([[[1, 0, 0]]]).float()

        self.attention(query, mask=mask)

        self.assertEqual((1, 1, 3), self.attention.scores.shape)

        q_expected = F.linear(query, self.W_Q, self.b_Q)
        k_expected = F.linear(inputs, self.W_K, self.b_K)

        a = torch.bmm(q_expected, k_expected.permute(0, 2, 1)) / np.sqrt(self.attention.d_model)
        expected_scores = torch.softmax(torch.tensor([[[a[:, :, 0].item(), -1e9, -1e9]]]).float(), dim=-1)

        torch.testing.assert_close(expected_scores, self.attention.scores)

if __name__ == '__main__':
    unittest.main()
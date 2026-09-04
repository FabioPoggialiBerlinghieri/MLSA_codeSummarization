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

    def test_init_state(self):
        # input : B x L x Input = 1 x 3 x 2
        inputs = torch.tensor([[[1,2],[2,-1],[-1,3]]]).float()

        self.attention.init_state(inputs)

        self.assertEqual((1, 3, 2), self.attention.k.shape)
        self.assertEqual((1, 3, 2), self.attention.v.shape)

        torch.testing.assert_close(F.linear(inputs, self.W_K, self.b_K), self.attention.k)
        torch.testing.assert_close(F.linear(inputs, self.W_V, self.b_V), self.attention.v)


    def test_init_state_with_wrong_input_dim(self):
        # wrong input
        wrong_inputs = torch.tensor([1,2]).float()

        with self.assertRaises(ValueError) as e:
            self.attention.init_state(wrong_inputs)

        self.assertEqual(f"Expected a 3D tensor (B, L, Input), received a {wrong_inputs.dim()}D tensor.",
                         str(e.exception))

        wrong_inputs = torch.tensor([[[1],[2],[-1]]]).float()
        with self.assertRaises(ValueError) as e:
            self.attention.init_state(wrong_inputs)

        self.assertEqual(f"Incorrect input dimension: expected 2, received {wrong_inputs.shape[2]}.",
                         str(e.exception))

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

    def test_forward_before_init_state_raises_error(self):
        query = torch.tensor([[[1.0, 0.0]]]).float()

        with self.assertRaises(RuntimeError) as e:
            self.attention(query)

        self.assertEqual(
            "Attention state not initialized. You must call 'init_state(inputs)' before calling 'forward(query)'.",
            str(e.exception))

    def test_forward_with_wrong_input_dim(self):
        # input : B x L x Input = 1 x 3 x 2
        inputs = torch.tensor([[[1, 2], [2, -1], [-1, 3]]]).float()
        self.attention.init_state(inputs)

        # rigth_query : B x 1 x Input

        wrong_query = torch.tensor([[1, 2]]).float()

        with self.assertRaises(ValueError) as e:
            self.attention(wrong_query)

        self.assertEqual(f"Expected query to be a 3D tensor (B, 1, Input), but received a {wrong_query.dim()}D tensor.",
                         str(e.exception))

        wrong_query = torch.tensor([[[1, 2], [0, 1]]]).float()

        with self.assertRaises(ValueError) as e:
            self.attention(wrong_query)

        self.assertEqual(f"Expected query sequence length to be 1 (B, 1, Input), but received {wrong_query.shape[1]} at dimension 1.",
                         str(e.exception))

        wrong_query = torch.tensor([[[1]]]).float()

        with self.assertRaises(ValueError) as e:
            self.attention(wrong_query)

        self.assertEqual(
            f"Expected query feature dimension to match input_dim (2), but received {wrong_query.shape[2]} at dimension 2.",
            str(e.exception))

    def test_forward_with_wrong_mask_dim(self):
        # input : B x L x Input = 1 x 3 x 2
        inputs = torch.tensor([[[1, 2], [2, -1], [-1, 3]]]).float()
        self.attention.init_state(inputs)

        # query : B x 1 x Input
        query = torch.tensor([[[1, 2]]]).float()

        # rigth mask : B x 1 x L
        wrong_mask = torch.tensor([[1, 0, 1]]).float()

        with self.assertRaises(ValueError) as e:
            self.attention(query, mask=wrong_mask)

        self.assertEqual(f"Expected mask to be a 3D tensor (B, 1, L), but received a {wrong_mask.dim()}D tensor.",
                         str(e.exception))

        wrong_mask = torch.tensor([[[1, 0, 1], [0, 1, 0]]]).float()

        with self.assertRaises(ValueError) as e:
            self.attention(query, mask=wrong_mask)

        self.assertEqual(
            f"Expected mask sequence length to be 1 (B, 1, L), but received {wrong_mask.shape[1]} at dimension 1.",
            str(e.exception))

        wrong_mask = torch.tensor([[[1, 0]]]).float()

        with self.assertRaises(ValueError) as e:
            self.attention(query, mask=wrong_mask)

        self.assertEqual(
            f"Mask sequence length must match input sequence length. Expected 3, but received {wrong_mask.shape[2]} at dimension 2.",
            str(e.exception))

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
import unittest
import torch
import torch.nn as nn
import transformer as ts

class AttentionTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_input_state_defaultInputDim(self):
        d_model = 3
        len_seq = 5
        expected_Wk = torch.randn(d_model, d_model)
        expected_Wv = torch.randn(d_model, d_model)

        print(expected_Wv)
        print(expected_Wk)

        self.attention = ts.Attention(d_model)

        x1 = torch.randn(len_seq, d_model)
        x2 = torch.randn(len_seq, d_model)
        batch = torch.Tensor([x1, x2])

        # B x L x Input = 2 x 5 x 3
        self.attention.init_state(batch)



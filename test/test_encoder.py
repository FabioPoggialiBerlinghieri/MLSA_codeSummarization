import unittest
import torch

from encoder import SelfAttentionEncoder


class EncoderTest(unittest.TestCase):

    def setUp(self) -> None:
        self.d_model = 3
        self.input_dim = 2
        self.input = torch.Tensor([[[2, 5], [6, 23], [0, 2940], [10, 11]]])
        self.encoder = SelfAttentionEncoder(self.d_model, self.input_dim)

    def test_forwardOutputDimension_noMask(self):
        output = self.encoder.forward(self.input)
        self.assertEqual(output.shape, (1, 4, 3))

    def test_forwardOutputDimension_withMask(self):
        mask = torch.Tensor([[[1,1,1,1], [1,1,1,1], [1,1,1,1], [0,0,1,0]]])
        output = self.encoder.forward(self.input, mask)
        self.assertEqual(output.shape, (1, 4, 3))

    def test_forwardOutputDimension_MoreBatch(self):
        b1 = self.input
        b2 = torch.Tensor([[[21, 9], [88, 0], [12, 12], [-2, -1]]])
        inputs = torch.cat((b1, b2), 0)
        output = self.encoder.forward(inputs)
        self.assertEqual(output.shape, (2, 4, 3))

if __name__ == '__main__':
    unittest.main()
import unittest
import torch
from encoder import SelfAttentionEncoder

class EncoderTest(unittest.TestCase):

    def setUp(self) -> None:
        self.d_model = 2
        self.input = torch.Tensor([[[2, 5], [6, 23], [0, 2940], [10, 11]]])
        self.mask = torch.Tensor([[[1,1,1,1], [1,1,1,1], [1,1,1,1], [0,0,1,0]]])
        self.encoder = SelfAttentionEncoder(self.d_model)

    def test_forwardOutputDimension(self):
        output = self.encoder.forward(self.input, self.mask)
        self.assertEqual(output.shape, (1, 4, 2))

    def test_forwardOutputDimension_MoreBatch(self):
        input2 = torch.Tensor([[[21, 9], [88, 0], [12, 12], [-2, -1]]])
        mask2 = torch.Tensor([[[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 0, 1], [0, 0, 1, 0]]])
        inputs = torch.cat((self.input, input2), 0)
        masks = torch.cat((self.mask, mask2), 0)
        output = self.encoder.forward(inputs, masks)
        self.assertEqual(output.shape, (2, 4, 2))

if __name__ == '__main__':
    unittest.main()
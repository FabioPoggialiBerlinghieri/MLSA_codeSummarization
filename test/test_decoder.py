import unittest
import torch
from decoder import Decoder

class DecoderTest(unittest.TestCase):

    def setUp(self) -> None:
        self.d_model = 3
        self.input_dim = 2
        self.output_dim = 2
        self.input = torch.Tensor([[[2, 5], [6, 23], [0, 2940], [10, 11]]])
        self.mask = torch.Tensor([[[1,1,1,1], [1,1,1,1], [1,1,1,1], [0,0,1,0]]])
        self.decoder = Decoder(self.d_model, self.input_dim, self.output_dim)

    def test_forwardOutputDimension(self):
        pass

    def test_forwardOutputDimension_MoreBatch(self):
        pass

if __name__ == '__main__':
    unittest.main()
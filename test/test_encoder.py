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

    def test_forwardOutputDimension_IncorrectInputDimension(self):
        self.input = torch.Tensor([[1, 2], [3, 4], [1, 1], [-3, 0]])
        self.assertRaisesRegex(ValueError,
                               r"Expected a 3D tensor\(B, L, Input\), received a 2D tensor\.",
                               lambda : self.encoder.forward(self.input))

    def test_forwardOutputDimension_InconsistentInputDimension(self):
        self.encoder.input_dim = 3
        self.assertRaisesRegex(ValueError,
                               "Incorrect input dimension: expected 3, received 2.",
                               lambda : self.encoder.forward(self.input))

    def test_forwardOutputDimension_IncorrectMask(self):
        incorrect_mask = torch.Tensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 1, 0]])
        self.assertRaisesRegex(ValueError,
                               r"Expected mask to be a 3D tensor \(B, L, L\), but received a 2D tensor\.",
                               lambda : self.encoder.forward(self.input, incorrect_mask))

    def test_forwardOutputDimension_InconsistentMask(self):
        incorrect_mask = torch.Tensor([[[1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]]])
        with self.assertRaises(ValueError) as e:
            self.encoder.forward(self.input, incorrect_mask)
        self.assertEqual(f"Expected mask to have shape (B, {self.input.shape[1]}, {self.input.shape[1]}), but received {incorrect_mask.shape}.",
                         str(e.exception))

        incorrect_mask = torch.Tensor([[[1, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 0]]])
        with self.assertRaises(ValueError) as e:
            self.encoder.forward(self.input, incorrect_mask)
        self.assertEqual(
            f"Expected mask to have shape (B, {self.input.shape[1]}, {self.input.shape[1]}), but received {incorrect_mask.shape}.",
            str(e.exception))


if __name__ == '__main__':
    unittest.main()
import unittest
import torch
from architecture.decoder import Decoder

class DecoderTest(unittest.TestCase):

    def setUp(self) -> None:
        self.d_model = 2

        self.encoder_state = torch.Tensor([[[2, 5], [6, 23], [0, 2940], [10, 11]]])
        self.input_mask = torch.Tensor([[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 0]]])

        self.labels = torch.Tensor([[[1, 2], [3, 4], [5, 6], [7, 8]]])
        self.labels_mask = torch.Tensor([[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]])

        self.decoder = Decoder(self.d_model)

    def test_forwardOutputDimension(self):
        self.decoder.init_state(self.encoder_state, self.input_mask)
        output = self.decoder.forward(self.labels, self.labels_mask)
        self.assertEqual(output.shape, (1, 4, 2))

    def test_forwardOutputDimension_MoreBatch(self):
        encoder_state2 = torch.Tensor([[[21, 9], [88, 0], [12, 12], [-2, -1]]])
        input_mask2 = torch.Tensor([[[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 0, 1], [0, 0, 1, 0]]])

        labels2 = torch.Tensor([[[-1, -2], [-3, -4], [-5, -6], [-7, -8]]])
        labels_mask2 = torch.Tensor([[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]])
        batch_encoder_state = torch.cat((self.encoder_state, encoder_state2), dim=0)
        batch_input_mask = torch.cat((self.input_mask, input_mask2), dim=0)

        batch_labels = torch.cat((self.labels, labels2), dim=0)
        batch_labels_mask = torch.cat((self.labels_mask, labels_mask2), dim=0)

        self.decoder.init_state(batch_encoder_state, batch_input_mask)

        output = self.decoder.forward(batch_labels, batch_labels_mask)

        self.assertEqual(output.shape, (2, 4, 2))

if __name__ == '__main__':
    unittest.main()
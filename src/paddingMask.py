import torch

class PaddingMask:

    @staticmethod
    def generate_padding_mask(x: torch.Tensor) -> torch.Tensor:
        mask = (x != 0).int()
        return mask.unsqueeze(1).repeat(1, x.size(1), 1).long()

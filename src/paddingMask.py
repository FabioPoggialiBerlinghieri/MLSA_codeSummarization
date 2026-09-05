import torch


class PaddingMask:

    @staticmethod
    def generate_mask(batch_size: int, seq_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        pos = torch.arange(max_length, device=seq_lengths.device)
        mask_2d = pos < seq_lengths.unsqueeze(1)
        mask_3d = mask_2d.unsqueeze(1).expand(-1, max_length, -1).float()
        return mask_3d
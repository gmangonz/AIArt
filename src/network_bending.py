import torch


class NetworkBending(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, transformation_dict: dict = {}) -> torch.Tensor:
        return x

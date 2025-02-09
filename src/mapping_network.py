import torch

from layers import FullyConnectedLayer


class Mapping(torch.nn.Module):
    def __init__(self, num_blocks: int, latent_dim: int, num_layers: int = 8):
        super(Mapping, self).__init__()

        self.num_blocks = num_blocks
        layers = []
        for _ in range(num_layers):
            layers.append(FullyConnectedLayer(in_features=latent_dim, out_features=latent_dim, lr_multiplier=0.01))
            layers.append(torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generates the latent vector w from the input z.
        The latent vector exists in latent space W that will be
        used to generate the style vectors for each layer.

        Parameters
        ----------
        z : torch.Tensor
            Tensor of shape [B, latent_dim].

        Returns
        -------
        torch.Tensor
            Tensor of shape [B, num_blocks, latent_dim].
        """

        z = torch.nn.functional.normalize(z, dim=1)
        w = self.net(z)
        w = w.unsqueeze(1).repeat([1, self.num_blocks, 1])
        return w

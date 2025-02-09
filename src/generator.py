import torch

from layers import FullyConnectedLayer, ModulatedConv, Upfirdn2d
from network_bending import NetworkBending


class toRGB(torch.nn.Module):
    def __init__(self, latent_dim: int, in_channels: int, **kwargs):
        super(toRGB, self).__init__(**kwargs)

        self.to_style = FullyConnectedLayer(
            in_features=latent_dim,
            out_features=in_channels,
            bias_init=1.0,
        )

        self.conv = ModulatedConv(in_features=in_channels, out_features=3, kernel_size=1, demod=False)
        self.bias = torch.nn.Parameter(torch.zeros(3), requires_grad=True)
        self.weight_gain = 1 / (in_channels * (1**2)) ** 0.5

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Generates an RGB image from a feature map using 1x1 convolution.

        Parameters
        ----------
        x : torch.Tensor
            Feature map tensor of shape [B, in_channels, H, W].
        w : torch.Tensor
            Latent vector tensor of shape [B, latent_dim].

        Returns
        -------
        torch.Tensor
            RGB image tensor of shape [B, 3, H, W].
        """

        style = self.to_style(w) * self.weight_gain  # Output shape will have the same number of features as x
        x = self.conv(x, style)
        x = x + self.bias[None, :, None, None]
        return torch.nn.functional.leaky_relu(x, 0.2, True)


class StyleBlock(torch.nn.Module):

    def __init__(self, latent_dim: int, in_channels: int, out_channels: int, up: bool = False, **kwargs):
        super(StyleBlock, self).__init__(**kwargs)
        kernel_size = 3

        self.to_style = FullyConnectedLayer(in_features=latent_dim, out_features=in_channels, bias_init=1.0)
        self.conv = ModulatedConv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            up=up,
        )
        self.bias = torch.nn.Parameter(torch.randn(out_channels), requires_grad=True)

        self.ScaleNoise = torch.nn.Parameter(torch.zeros(1))
        self.activation = torch.nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Passes latent vector w through affine transformation (self.to_style). (A)
        Then modulates the styles with the conv weights. Demodulates the weights with 1/sigma.
        Passes feature map through a 3x3 convolution with demod weights.
        Adds scaled noise and bias to the output.
        Run result through activation function.

        Parameters
        ----------
        x : torch.Tensor
            Feature map tensor of shape [B, in_channels, H, W].
        w : torch.Tensor
            Latent vector tensor of shape [B, latent_dim].
        noise : torch.Tensor
            Input noise tensor of shape [B, 1, 2**res, 2**res]

        Returns
        -------
        torch.Tensor
            Feature map tensor of shape [B, out_channels, H, W].
        """

        s = self.to_style(w)  # Output shape will have the same number of features as x
        x = self.conv(x, s)
        x = x + self.ScaleNoise[None, :, None, None] * noise  # Broadcast to x's shape
        x = x + self.bias[None, :, None, None]
        return self.activation(x)


class GeneratorBlock(torch.nn.Module):

    def __init__(self, latent_dim: int, in_channels: int, out_channels: int, **kwargs):
        # self.layer_name = self.name

        super(GeneratorBlock, self).__init__(**kwargs)
        self.style_block1 = StyleBlock(latent_dim, in_channels, out_channels, up=True)
        self.style_block2 = StyleBlock(latent_dim, out_channels, out_channels)

        # Add network bending
        self.networkbending = NetworkBending()

        self.to_rgb = toRGB(latent_dim=latent_dim, in_channels=out_channels)

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, noise: tuple[torch.Tensor, torch.Tensor], transformation_dict: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The individual block of the StyleGAN2 Generator that:
            Takes image and upsamples x2 (in self.style_block1 within ModulatedConv(...)),
            Does twice: Takes w vector, performs affine transformation (self.to_style()), performs modulated 3x3 convolution, adds scaled noise and bias.
            Takes output and converts it to a 3-channel RGB image.

        Parameters
        ----------
        x : torch.Tensor
            Feature map tensor of shape [B, in_channels, H, W].
        w : torch.Tensor
            Latent vector tensor of shape [B, latent_dim].
        noise : tuple[torch.Tensor, torch.Tensor]
            Noise tensor of shape [2, B, 2**res, 2**res, 1].
        transformation_dict : dict, optional
            Mapping of name to transformation, by default {}.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Feature map of shape [B, out_channels, H, W] and RGB image of shape [B, 3, H, W].
        """

        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])
        rgb_img = self.to_rgb(x, w)
        rgb_img = self.networkbending(rgb_img, transformation_dict)
        # Feature map, rgb image
        return x, rgb_img


class Generator(torch.nn.Module):
    def __init__(self, max_log2_res, latent_dim=512, min_num_features=32, max_num_features=512, **kwargs):
        super(Generator, self).__init__(**kwargs)

        # [min(512, 32 * (2 ** i)) for i in range(10 - 2, -1, -1)] -> [512, 512, 512, 512, 512, 256, 128, 64, 32]
        out_channels_list = [min(max_num_features, min_num_features * (2**i)) for i in range(max_log2_res - 2, -1, -1)]

        # First block takes 4x4 constant input and only has 1 StyleBlock rather than 2.
        self.style_block_1 = StyleBlock(
            latent_dim=latent_dim, in_channels=out_channels_list[0], out_channels=out_channels_list[0]
        )
        self.to_rgb_1 = toRGB(latent_dim=latent_dim, in_channels=out_channels_list[0])

        # Generate the Remaining Generator Blocks
        self.number_of_blocks = len(out_channels_list)  # Should be max_log2_res - 1
        self.generator_blocks = torch.nn.ModuleList(
            [
                GeneratorBlock(
                    latent_dim=latent_dim, in_channels=out_channels_list[i - 1], out_channels=out_channels_list[i]
                )
                for i in range(1, self.number_of_blocks)
            ]
        )

        upx, upy = 2, 2
        px0, px1, py0, py1 = 0, 0, 0, 0
        resample_filter = [1, 3, 3, 1]
        fw, fh = len(resample_filter), len(resample_filter)

        self.up_sample = Upfirdn2d(
            resample_filter=resample_filter,
            up=2,
            padding=[
                px0 + (fw + upx - 1) // 2,
                px1 + (fw - upx) // 2,
                py0 + (fh + upy - 1) // 2,
                py1 + (fh - upy) // 2,
            ],
            gain=upx * upy,
        )

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, noise: torch.Tensor, transformation_dict: dict = {}
    ) -> torch.Tensor:
        """
        Image Generator.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape [B, max_num_features, 4, 4].
        w : torch.Tensor
            Style inputs: [B, number_of_blocks, 512].
        noise : torch.Tensor
            Noise tensor of shape [[2, B, 1, 2**res, 2**res] for res in range(2, log2_end_res+1)].
        transformation_dict : dict, optional
            Mapping of name to transformation, by default {}.

        Returns
        -------
        torch.Tensor
            Output image tensor of shape [B, 3, 2**log2_end_res, 2**log2_end_res].
        """

        x = self.style_block_1(x, w[:, 0, :], noise[0][0])
        rgb = self.to_rgb_1(x, w[:, 0, :])

        # Start at 1 since noise[0] and w[:, 0, :] have already been
        # used, but that means use i-1 as index for generator_blocks list
        for i in range(1, self.number_of_blocks):
            x, newRGB = self.generator_blocks[i - 1](x, w[:, i, :], noise[i], transformation_dict)
            rgb = newRGB + self.up_sample(rgb)

        return rgb

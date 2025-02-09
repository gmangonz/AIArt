import torch

from layers import Conv2dLayer, FullyConnectedLayer


class MiniBatchSTD(torch.nn.Module):
    def __init__(self, num_new_features: int = 1, **kwargs):

        super(MiniBatchSTD, self).__init__(**kwargs)
        self.num_new_features = num_new_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        group_size = min(4, B)
        y = x.reshape(
            [group_size, -1, self.num_new_features, C // self.num_new_features, H, W]
        )  # [G, M, n, c, H, W] Split batch into M groups of size G. Split channels into n channel groups c.

        y = y.to(torch.float32)  # [G, M, n, c, H, W] Cast to FP32.
        y = y - torch.mean(y, dim=0, keepdims=True)  # [G, M, n, c, H, W] Subtract mean over group.
        y = torch.square(y).mean(dim=0)  # [M, n, c, H, W]  Calc variance over group.
        y = torch.sqrt(y + 1e-8)  # [M, n, c, H, W]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4], keepdim=True)  # [M, n, 1, 1, 1]  Take average over fmaps and pixels.
        y = y.mean(dim=[2])  # [M, n, 1, 1]   Split channels into c channel groups.
        y = torch.tile(y, [group_size, 1, H, W])  # [B, n, H, W]   Replicate over group and pixels.

        return torch.cat([x, y], dim=1)  # [B, C, H, W]   Append to input as new channels.


class DiscriminatorBlock(torch.nn.Module):
    """
    Residual connected discriminator block.

    In main branch: Perform 2 3x3 convolutions and downsample.
    In residual branch: Downsample and then perform 1x1 convolution.

    Add the two branches.
    """

    def __init__(self, in_features: int, out_channels: int, **kwargs):

        super(DiscriminatorBlock, self).__init__(**kwargs)
        activation = "lrelu"

        self.skip = Conv2dLayer(
            in_features,
            out_channels,
            kernel_size=1,
            bias=False,
            down=2,
        )

        self.block = torch.nn.Sequential(
            Conv2dLayer(
                in_features,
                in_features,
                kernel_size=3,
                activation=activation,
            ),
            Conv2dLayer(
                in_features,
                out_channels,
                kernel_size=3,
                activation=activation,
                down=2,
            ),
        )
        self.scale = 1 / (2**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.skip(x)
        x = self.block(x)
        return (x + y) * self.scale


class Discriminator(torch.nn.Module):

    def __init__(self, max_log2_res: int = 10, min_num_features: int = 64, max_num_features: int = 512, **kwargs):
        super().__init__(**kwargs)

        # [64, 128, 256, 512, 512, 512, 512, 512] (max_log2_res=9)
        features = [min(max_num_features, min_num_features * (2**i)) for i in range(max_log2_res - 1)]
        num_blocks = len(features) - 1  # -1 because the last 4x4 will be performed outside

        # Converts RGB image to a feature map
        self.fromRGB = Conv2dLayer(
            in_features=3,
            out_features=min_num_features,
            kernel_size=1,
            bias=True,
            activation="lrelu",
        )

        # Discriminator blocks
        self.blocks = torch.nn.Sequential(
            *[DiscriminatorBlock(features[i], features[i + 1]) for i in range(num_blocks)]
        )

        mbstd_num_channels = 1
        self.std_dev = MiniBatchSTD(num_new_features=mbstd_num_channels)

        final_features = features[-1]
        self.conv = torch.nn.Sequential(
            *[
                Conv2dLayer(
                    final_features + mbstd_num_channels, out_features=final_features, kernel_size=3, activation="lrelu"
                ),
                torch.nn.Flatten(),
                torch.nn.Dropout(0.5),
            ]
        )

        # Final linear layer to get the classification
        self.fcl = FullyConnectedLayer(
            in_features=4 * 4 * final_features, out_features=final_features, activation="lrelu"
        )
        self.out = FullyConnectedLayer(in_features=final_features, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # From RGB (first block)
        x = self.fromRGB(x)
        # Skip connection blocks
        x = self.blocks(x)
        # Epilogue
        x = self.std_dev(x)  # [1, 512, 4, 4] -> [1, 513, 4, 4]
        x = self.conv(x)  # [1, 513, 4, 4] -> [1, 8192]
        x = self.fcl(x)  # [1, 8192] -> [1, 512]
        return self.out(x)

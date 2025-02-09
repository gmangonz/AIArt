import torch


class Conv2dLayer(torch.nn.Module):
    """
    Used mainly within the Discriminator class and sub-blocks, where only downsampling occurs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        bias: bool = True,
        bias_init: float = 0.0,
        activation: str = "linear",
        down=1,
    ):
        super(Conv2dLayer, self).__init__()
        weight_shape = [out_features, in_features, kernel_size, kernel_size]
        self.activation = activation

        self.weight = torch.nn.Parameter(torch.randn(weight_shape))
        self.bias = torch.nn.Parameter(torch.ones([out_features]) * bias_init) if bias else None
        self.gain = 1 / ((in_features * kernel_size**2) ** 0.5)

        self.__build_downsample(kernel_size=kernel_size, down=down)

    def __build_downsample(self, kernel_size: int, down: int):

        conv2d_padding = 0
        conv2d_stride = 1

        resample_filter = [1, 3, 3, 1]
        fw, fh = len(resample_filter), len(resample_filter)
        padding = kernel_size // 2
        px0, px1, py0, py1 = [padding] * 4

        if down > 1:
            px0 += (fw - down + 1) // 2
            px1 += (fw - down) // 2
            py0 += (fh - down + 1) // 2
            py1 += (fh - down) // 2

        if kernel_size == 1 and (down > 1):
            self.resample = Upfirdn2d(resample_filter=resample_filter, padding=[px0, px1, py0, py1], down=down)
        elif down > 1:
            self.resample = Upfirdn2d(resample_filter=resample_filter, padding=[px0, px1, py0, py1])
            conv2d_stride = 2
        elif down == 1:
            self.resample = lambda x: x
            conv2d_padding = [py0, px0]
        else:
            raise NotImplementedError()
        self.conv_kwargs = dict(stride=conv2d_stride, padding=conv2d_padding)

    def forward(self, x: torch.Tensor):

        w = self.weight * self.gain
        b = self.bias.to(x.dtype) if self.bias is not None else None

        x = self.resample(x)
        x = torch.nn.functional.conv2d(x, w, bias=b, groups=1, **self.conv_kwargs)

        if self.activation == "lrelu":
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
        return x


class FullyConnectedLayer(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "linear",
        lr_multiplier: float = 1.0,
        bias_init: float = 0.0,
    ):
        super(FullyConnectedLayer, self).__init__()
        _shape = [out_features, in_features]

        self.weight = torch.nn.Parameter(torch.randn(_shape) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.ones([out_features]) * bias_init)
        self.gain = lr_multiplier / in_features**0.5

        self.bias_gain = lr_multiplier
        self.activation = activation

    def forward(self, x: torch.Tensor):
        w = self.weight * self.gain
        b = self.bias * self.bias_gain

        if self.activation == "linear":
            return torch.addmm(b.unsqueeze(0), x, w.t())

        elif self.activation == "lrelu":
            return torch.nn.functional.leaky_relu(
                input=torch.nn.functional.linear(input=x, weight=w, bias=b),
                negative_slope=0.2,
                inplace=True,
            )


class ModulatedConv(torch.nn.Module):
    """
    Used mainly within the Generator class and sub-blocks, where only upsampling occurs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        padding: int = 0,
        demod: bool = True,
        fused_modconv: bool = True,
        resample_kernel=None,
        lr_multiplier: float = 1.0,
        up: bool = False,
    ):
        super(ModulatedConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.padding = padding

        self.demod = demod
        self.fused_modconv = fused_modconv
        self.resample_kernel = resample_kernel
        self.lr_multiplier = lr_multiplier

        self.up = up
        self.__build()

    def __build(self):

        kernel_shape = [
            self.out_features,
            self.in_features,
            self.kernel_size,
            self.kernel_size,
        ]  # (out_channels, in_channels, kh, kw)
        self.gain: float = self.lr_multiplier / ((self.out_features * self.in_features * self.kernel_size**2) ** 0.5)

        self.w = torch.nn.Parameter(
            torch.randn(*kernel_shape) / self.lr_multiplier
        )  # Initialize with mean=0, std=1 / lr_multiplier

        self.__build_upsample()

    def __build_upsample(self):

        if self.up:
            resample_filter = [1, 3, 3, 1]
            fw, fh = len(resample_filter), len(resample_filter)
            px0, px1, py0, py1 = [self.padding] * 4 if isinstance(self.padding, int) else self.padding
            px0 += (fw + 2 - 1) // 2
            px1 += (fw - 2) // 2
            py0 += (fh + 2 - 1) // 2
            py1 += (fh - 2) // 2

            if self.kernel_size == 1:
                self.resample = Upfirdn2d(
                    resample_filter=resample_filter, up=2, padding=[px0, px1, py0, py1], gain=2**2
                )
                self.conv_kwargs = dict()
            else:
                px0 -= self.kernel_size - 1
                px1 -= self.kernel_size - 2
                py0 -= self.kernel_size - 1
                py1 -= self.kernel_size - 2
                pxt = max(min(-px0, -px1), 0)
                pyt = max(min(-py0, -py1), 0)
                self.resample = Upfirdn2d(
                    resample_filter=resample_filter,
                    padding=[px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt],
                    gain=2**2,
                )
                self.conv_kwargs = dict(stride=2, padding=[pyt, pxt])

    def conv2d_resample(self, x: torch.Tensor, w: torch.Tensor, groups: int) -> torch.Tensor:

        if self.up:
            if self.kernel_size == 1:
                x = torch.nn.functional.conv2d(x, w, groups=groups, **self.conv_kwargs)
            else:
                if groups == 1:
                    w = w.transpose(0, 1)
                else:
                    out_channels, in_channels, kh, kw = w.shape
                    w = w.reshape(groups, out_channels // groups, in_channels, kh, kw)
                    w = w.transpose(1, 2)
                    w = w.reshape(groups * in_channels, out_channels // groups, kh, kw)
                x = torch.nn.functional.conv_transpose2d(x, w, groups=groups, **self.conv_kwargs)
            x = self.resample(x)
        else:
            x = torch.nn.functional.conv2d(
                x,
                w,
                stride=1,
                padding=self.padding,
                groups=groups,
            )
        return x

    def forward(self, x: torch.Tensor, styles: torch.Tensor):  # x: (N, C, H, W), styles: (N, C)

        assert x.shape[0] == styles.shape[0], "x and styles must have the same batch size"
        assert styles.shape[1] == x.shape[1] == self.in_features, "Mismatch in input dimensions"

        batch_size, _, height, width = x.shape

        if self.demod or self.fused_modconv:
            w = self.w[None, ...] * self.gain  # (1, out_c, in_c, kh, kw)
            w = w * (
                styles[:, None, :, None, None] + 1
            )  # reshapes styles to (N, 1, in_c, 1, 1) -> w is (N, out_c, in_c, kh, kw)
        if self.demod:
            dcoefs = torch.rsqrt((w**2).sum(dim=[2, 3, 4]) + 1e-8)  # (N, out_c)
            if self.fused_modconv:
                w = w * dcoefs[:, :, None, None, None]  # (N, out_c, in_c, kh, kw)

        if not self.fused_modconv:
            x = x * styles[:, :, None, None]  # Modulation: (N, in_c, H, W) * (N, in_c, 1, 1) -> (N, in_c, H, W)
            w = self.w  # [out_c, in_c, kh, kw]
        else:
            x = x.view(1, -1, height, width)  # (N, in_c, H, W) -> (1, N*in_c, H, W)
            _, _, *ws = w.shape  # (N, out_c, in_c, kh, kw)
            w = w.reshape(-1, *ws)  # (N*out_c, in_c, kh, kw)

        # Convolution with optional upsampling.
        x = self.conv2d_resample(x, w, groups=batch_size if self.fused_modconv else 1)

        if self.fused_modconv:
            x = x.view(batch_size, self.out_features, x.shape[2], x.shape[3])  # Reshape back to (N, -1, H', W')
        elif self.demod and not self.fused_modconv:
            x = x * dcoefs[:, :, None, None]  # Demodulation: (N, out_c, H', W') * (N, out_c, 1, 1)
        return x


class Upfirdn2d(torch.nn.Module):

    def __init__(
        self,
        resample_filter: list = [1, 3, 3, 1],
        up: int = 1,
        down: int = 1,
        padding: int = 0,
        gain: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert isinstance(resample_filter, list)

        # Setup filter
        filter = torch.as_tensor(resample_filter, dtype=torch.float32)
        filter = filter.ger(filter)  # Outer product
        filter /= filter.sum()
        self.register_buffer("filter", filter)  # Setup the filter

        self.up = up
        self.down = down
        self.padding = [padding] * 4 if isinstance(padding, int) else padding
        assert isinstance(self.padding, list)
        assert len(self.padding) == 4

        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get relevant shapes
        batch, channels, in_height, in_width = x.shape
        upx, upy = self.up, self.up
        padx0, padx1, pady0, pady1 = self.padding
        # Interleave dimensions
        x = x.view(batch, channels, in_height, 1, in_width, 1)
        x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
        x = x.view(batch, channels, in_height * upy, in_width * upx)
        # Add padding/cropping
        x = torch.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
        x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]
        # Scale and reshape filter
        assert self.filter.ndim == 2
        f: torch.Tensor = self.filter * (self.gain ** (self.filter.ndim / 2))
        f = f.to(x.dtype)
        f = f[None, None].repeat([channels, 1, 1, 1])
        # Apply convolution for FIR filtering
        output = torch.nn.functional.conv2d(input=x, weight=f, groups=channels)
        output = output[:, :, :: self.down, :: self.down]
        return output

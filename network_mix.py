from torch import nn

class Conv(nn.Module):
    """A convolutional operation with ReLU activation and batch normalization.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolution kernel.
        stride (int): The stride of the convolution.
        padding (int): The padding applied to the input.
        affine (bool): Whether to include learnable affine parameters in the
                       batch normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 affine=True):
        super(Conv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """A separable convolutional operation with ReLU activation and batch
       normalization.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolution kernel.
        stride (int): The stride of the convolution.
        padding (int): The padding applied to the input.
        affine (bool): Whether to include learnable affine parameters in the
                       batch normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=in_channels,
                      bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channels, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class NetworkMixArch(nn.Module):
    """A neural network architecture based on MixNet.

    Args:
        channels (int): Number of channels in the first layer of the network.
        num_classes (int): Number of output classes.
        layers (int): Number of layers in the network.
        mixnet_code (list[int]): Code indicating which type of layer to use at
                                 each depth.
        kernel_sizes (list[int]): Size of the kernel to use for each layer.
        image_size (tuple[int, int]): Size of the input image.
    """

    def __init__(self, channels, num_classes, layers, mixnet_code,
                 kernel_sizes, image_size):
        super(NetworkMixArch, self).__init__()
        self._layers = layers
        # Set stride to 2 for first layers based on image size
        strides = [1, 1, 1]
        for i in range(image_size[0] // 64):
            strides[i] = 2
        # Define stem layers
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels // 2, kernel_size=3, stride=strides[0],
                      padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=3,
                      stride=strides[1], padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=strides[2],
                      padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        # Define mix layers
        self.mix_layers = nn.ModuleList()
        prev_channels, curr_channels = channels, channels
        # reduction_prev = False
        for i in range(layers):
            # Determine whether to reduce feature map size
            if i in [layers // 3, 2 * layers // 3]:
                curr_channels *= 2
                reduction = True
            else:
                reduction = False
            # Set stride based on whether reduction is needed
            stride = 2 if reduction else 1
            # Set padding based on kernel size
            if kernel_sizes[i] == 3:
                pad = 1
            elif kernel_sizes[i] == 5:
                pad = 2
            else:
                pad = 3
            # Create mix layer
            if mixnet_code[i] == 0:
                mix_layer = SepConv(prev_channels, curr_channels,
                                    kernel_size=kernel_sizes[i], stride=stride,
                                    padding=pad, affine=True)
            else:
                mix_layer = Conv(prev_channels, curr_channels,
                                 kernel_size=kernel_sizes[i],
                                 stride=stride, padding=pad, affine=True)
            # Update previous channel count and reduction status
            # reduction_prev = reduction
            prev_channels = curr_channels
            self.mix_layers.append(mix_layer)
        # Define classifier layers
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(prev_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for mix_layer in self.mix_layers:
            x = mix_layer(x)
        out = self.global_pooling(x)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

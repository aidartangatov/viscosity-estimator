from quannet.tasks import BaseModel

import torch
import torch.nn as nn


class CNN3DModule(BaseModel):
    N_FILTERS = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.convnet = nn.Sequential(*[layer for block in self._set_conv_blocks(self.N_FILTERS) for layer in block])
        self.drop_out = nn.Dropout(0.05)
        self.fc = nn.Sequential(nn.Linear(self.N_FILTERS * 512, 1), nn.ReLU())

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel_size=3, padding='same', dilation=1, with_pooling=True):
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation), nn.ReLU()]
        if with_pooling:
            layers.append(nn.MaxPool3d(2))
        return layers

    def _set_conv_blocks(self, n_filters):
        convolution_blocks = [
            # No pooling included
            self._conv_block(2, 1, 1, with_pooling=False),
            # Pooling included
            self._conv_block(1, n_filters),
            # Each subsequent convolutional layer doubles the number of channels.
            # Input: from n_filters*1 to n_filters*16; output: from n_filters*2 to v*32
            *(self._conv_block(n_filters * (2**i), n_filters * (2 ** (i + 1)), with_pooling=True) for i in range(5)),
            # Last convolutional layer has 512 output channels
            self._conv_block(n_filters * 32, n_filters * 512, with_pooling=True),
        ]

        if self.args.grid_dim < 129:
            del convolution_blocks[0]  # Remove the first convolution block
            del convolution_blocks[-1][-1]  # Remove pooling for the last convolution layer
            if self.args.grid_dim < 64:
                del convolution_blocks[-2][-1]  # Remove pooling for the second to last convolution layer

        return convolution_blocks

    def forward(self, x):
        x = self.convnet(x)
        x = torch.flatten(x, 1)
        x = self.drop_out(x)
        x = self.fc(x)

        return x

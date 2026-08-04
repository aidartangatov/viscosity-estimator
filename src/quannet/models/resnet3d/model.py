from typing import Sequence
from quannet.tasks import BaseModel

import torch
import torch.nn as nn


class Projection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout_p: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout2 = nn.Dropout(dropout_p)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = Projection(in_channels, out_channels, stride=stride)
        else:
            self.shortcut = nn.Identity()
        self.act2 = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.bn2(self.conv2(x))
        x = self.dropout2(x)
        return self.act2(x + shortcut)


class ResNet3DModule(BaseModel):
    def __init__(
        self,
        n_blocks: Sequence[int] = (2, 2, 2, 2),
        n_channels: Sequence[int] = (4, 8, 16, 32),
        first_kernel_size: int = 3,
    ):
        super().__init__()
        assert len(n_blocks) == len(n_channels)
        self.conv = nn.Conv3d(1, n_channels[0], kernel_size=first_kernel_size, stride=2, padding=first_kernel_size // 2)
        self.bn = nn.BatchNorm3d(n_channels[0])

        blocks = []
        prev_channels = n_channels[0]
        for i, channels in enumerate(n_channels):
            stride = 2 if len(blocks) == 0 else 1
            blocks.append(ResBlock(prev_channels, channels, stride=stride))
            prev_channels = channels
            for _ in range(n_blocks[i] - 1):
                blocks.append(ResBlock(channels, channels, stride=1))
        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(0.1)
        # No ReLU on the regression head — see cnn3d/model.py for rationale.
        self.fc = nn.Sequential(nn.Linear(n_channels[-1], 1))

    def forward_spatial_features(self, x):
        """Encoder trunk: return the pre-pool spatial feature map (B, C, d, h, w).

        Used by the 3-D MAE pretext task, which needs spatial layout to
        reconstruct masked voxels; ``forward_features`` pools this down to a
        single vector for VICReg/linear-probing/the regression head.
        """
        x = self.bn(self.conv(x))
        x = self.blocks(x)
        return x

    def forward_features(self, x):
        """Encoder: return the pooled feature vector (B, C) before the head.

        This is the representation used for SSL pretraining and linear probing;
        ``forward`` just adds dropout + the regression head on top.
        """
        x = self.forward_spatial_features(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.mean(dim=-1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet3DModule(nn.Module):
    def __init__(self, size):
        super().__init__()
        block = ResBlock
        num_blocks = [2, 2, 2, 2]
        self.in_channels = 4

        self.conv1 = nn.Conv3d(1, 4, kernel_size=1, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(4)
        self.layer1 = self._make_layer(block, 4, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 8, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 16, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 32, num_blocks[3], stride=1)
        self.linear = nn.Sequential(nn.Linear(32, 1), nn.ReLU())

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool3d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = nn.Dropout(0.1)(out)
        x = self.linear(out)
        if y is not None:
            loss = nn.functional.huber_loss(y, x, reduction='mean')
            return x, loss
        else:
            return x

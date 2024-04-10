import torch
import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )

    def forward(self, x) -> Tensor:
        return self.conv(x)


class BlockWithoutPooling(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=(5, 5),
                stride=1,
                padding="same",
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x) -> Tensor:
        return self.conv(x)


class FCLayer(nn.Module):
    def __init__(self, in_channel: int, out_channel: int = 10) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channel, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x) -> Tensor:
        return self.fc(x)


# Build Network
class Network(nn.Module):
    def __init__(self, image_channels: int = 3, num_features: int = 32):
        super(Network, self).__init__()
        self.initial_block = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_features,
                padding=3,
                kernel_size=7,
                stride=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.middle_blocks = nn.ModuleList(
            [
                ConvBlock(in_channel=num_features, out_channel=num_features * 2),
                ConvBlock(in_channel=num_features * 2, out_channel=num_features * 4),
                ConvBlock(in_channel=num_features * 4, out_channel=num_features * 8),
                BlockWithoutPooling(
                    in_channel=num_features * 8, out_channel=num_features * 8
                ),
                BlockWithoutPooling(
                    in_channel=num_features * 8, out_channel=num_features * 8
                ),
                BlockWithoutPooling(
                    in_channel=num_features * 8, out_channel=num_features * 8
                ),
                BlockWithoutPooling(
                    in_channel=num_features * 8, out_channel=num_features * 8
                ),
                nn.Flatten(),
            ]
        )
        self.fc = FCLayer(in_channel=256 * 28 * 28, out_channel=10)

    def forward(self, x):
        x = self.initial_block(x)
        for layer in self.middle_blocks:
            x = layer(x)
        return self.fc(x)


def test():
    net = Network()
    input = torch.randn(1, 3, 224, 224)
    output = net.forward(input)
    print(output.shape)
    print(output)


if __name__ == "__main__":
    test()

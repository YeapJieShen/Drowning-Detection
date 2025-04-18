import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class CNNClassifier(nn.Module):
    def __init__(self, num_classes, num_blocks, first_out_channel=32, out_channel_multiplier=2, kernel_size=3, stride=1, padding=1, input_shape=(3, 128, 128)):
        super(CNNClassifier, self).__init__()

        C, H, W = input_shape

        self.features = nn.Sequential(
            ConvBlock(C, first_out_channel, kernel_size, stride, padding),
            *[
                ConvBlock(first_out_channel * (out_channel_multiplier ** i), first_out_channel *
                          (out_channel_multiplier ** (i + 1)), kernel_size, stride, padding)
                for i in range(0, num_blocks - 1)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(first_out_channel * (out_channel_multiplier ** (num_blocks - 1))
                      * (H // (2 ** num_blocks)) * (W // (2 ** num_blocks)), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, prob=False):
        x = self.features(x)
        x = self.classifier(x)

        if prob:
            x = nn.Softmax(dim=1)(x)

        return x

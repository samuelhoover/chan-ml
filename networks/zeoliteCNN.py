import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self, input_size):
        super(ConvNet, self).__init__()
        in_channel = input_size[0]
        self.adamaxpool_in = nn.AdaptiveMaxPool3d(output_size=input_size)
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel * 2, in_channel * 3, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channel * 3, in_channel * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(1152, 384),
            nn.LeakyReLU(),
            nn.Linear(384, 192),
            nn.LeakyReLU(),
            nn.Linear(192, 96),
            nn.LeakyReLU(),
            nn.Linear(96, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.adamaxpool_in(x)
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x


class ConvNetV2(nn.Module):

    def __init__(self, input_size):
        super(ConvNetV2, self).__init__()
        in_channel = input_size[0]
        self.adamaxpool_in = nn.AdaptiveMaxPool3d(output_size=input_size)
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel * 2, in_channel * 3, kernel_size=5, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(3456, 150),
            nn.ReLU(),
            nn.Linear(150, 1),
        )

    def forward(self, x):
        x = self.adamaxpool_in(x)
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x


class TinyConvNet(nn.Module):

    def __init__(self, input_size, output_size):
        super(TinyConvNet, self).__init__()
        in_channel = input_size[0]
        self.linear_input_size = output_size[0] * output_size[1] * output_size[2]  # input size of first linear layer
        self.adamaxpool_in = nn.AdaptiveMaxPool3d(output_size=input_size)
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(self.linear_input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.adamaxpool_in(x)
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x


class FCNN(nn.Module):

    def __init__(self, input_size):
        super(FCNN, self).__init__()
        in_channel = input_size[0]
        self.adamaxpool_in = nn.AdaptiveMaxPool3d(output_size=input_size)
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel * 2, in_channel * 4, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel * 4, in_channel * 6, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel * 6, in_channel * 8, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channel * 8, in_channel * 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channel * 8, in_channel * 8, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channel * 8, in_channel * 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channel * 8, in_channel * 8, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channel * 8, in_channel * 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channel * 8, in_channel * 8, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channel * 8, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.adamaxpool_in(x)
        x = self.features(x)
        x = self.classifier(x).view(-1)

        return x


class ConvNet3D(nn.Module):

    def __init__(self, input_size):
        super(ConvNet3D, self).__init__()
        in_channel = input_size[0]
        self.adamaxpool_in = nn.AdaptiveMaxPool3d(output_size=input_size)
        # self.conv1 = nn.Conv3d(1, in_channel, kernel_size=3, stride=2, padding=1)
        # self.conv2 = nn.Conv3d(in_channel, in_channel * 2, kernel_size=1, stride=2)
        # self.conv3 = nn.Conv3d(in_channel * 2, in_channel * 4, kernel_size=1, stride=2)
        # self.conv4 = nn.Conv3d(in_channel * 4, in_channel * 8, kernel_size=1, stride=2)
        # self.fc1 = nn.Linear(1536, 1)
        self.features = nn.Sequential(
            nn.Conv3d(1, in_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # nn.LayerNorm(),
            nn.Conv3d(in_channel, in_channel * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # nn.LayerNorm(),
            nn.Conv3d(in_channel * 2, in_channel * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # nn.LayerNorm(),
            nn.Conv3d(in_channel * 4, in_channel * 8, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # nn.LayerNorm(),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(8192, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.adamaxpool_in(x)
        x = torch.unsqueeze(x, dim=0)
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x).view(-1)

        return x

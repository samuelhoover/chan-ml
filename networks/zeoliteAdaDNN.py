import torch.nn as nn


class AdaSixLayerNet(nn.Module):

    def __init__(self, input_size):
        super(AdaSixLayerNet, self).__init__()
        self.linear_input_size = input_size[0] * input_size[1] * input_size[2]  # input size of first linear layer
        self.adamaxpool = nn.AdaptiveMaxPool3d(output_size=input_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.linear_input_size, 10000, bias=True), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(10000, 5000, bias=True), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(5000, 5000, bias=True), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(5000, 1000, bias=True), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(1000, 1000, bias=True), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(1000, 1, bias=True),
        )

    def forward(self, x):
        x = self.adamaxpool(x)
        x = x.view(-1, self.linear_input_size)
        x = self.classifier(x)
        return x


class AdaFourLayerNet(nn.Module):

    def __init__(self, input_size):
        super(AdaFourLayerNet, self).__init__()
        self.linear_input_size = input_size[0] * input_size[1] * input_size[2]  # input size of first linear layer
        self.adamaxpool = nn.AdaptiveMaxPool3d(output_size=input_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.linear_input_size, 10000, bias=True), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(10000, 5000, bias=True), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(5000, 1000, bias=True), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(1000, 1, bias=True),
        )

    def forward(self, x):
        x = self.adamaxpool(x)
        x = x.view(-1, self.linear_input_size)
        x = self.classifier(x)
        return x


class AdaThreeLayerNet(nn.Module):

    def __init__(self, input_size):
        super(AdaThreeLayerNet, self).__init__()
        self.linear_input_size = input_size[0] * input_size[1] * input_size[2]  # input size of first linear layer
        self.adamaxpool = nn.AdaptiveMaxPool3d(output_size=input_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.linear_input_size, 100, bias=True), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(50, 10, bias=True), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(10, 1, bias=True),
        )

    def forward(self, x):
        x = self.adamaxpool(x)
        x = x.view(-1, self.linear_input_size)
        x = self.classifier(x)
        return x


class AdaTwoLayerNet(nn.Module):

    def __init__(self, input_size):
        super(AdaTwoLayerNet, self).__init__()
        self.linear_input_size = input_size[0] * input_size[1] * input_size[2]  # input size of first linear layer
        self.adamaxpool = nn.AdaptiveMaxPool3d(output_size=input_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.linear_input_size, 50, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(50, 1, bias=True),
        )

    def forward(self, x):
        x = self.adamaxpool(x)
        x = x.view(-1, self.linear_input_size)
        x = self.classifier(x)
        return x


class AdaSVR(nn.Module):

    def __init__(self, input_size):
        super(AdaSVR, self).__init__()
        self.linear_input_size = input_size[0] * input_size[1] * input_size[2]  # input size of first linear layer
        self.adamaxpool = nn.AdaptiveMaxPool3d(output_size=input_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.linear_input_size, 1, bias=True),
        )

    def forward(self, x):
        x = self.adamaxpool(x)
        x = x.view(-1, self.linear_input_size)
        x = self.classifier(x)
        return x

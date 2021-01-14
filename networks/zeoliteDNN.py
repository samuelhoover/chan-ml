import torch.nn as nn

from utils.ops import Swish, LearnedSwish


class LongNet_V1(nn.Module):

    def __init__(self):
        super(LongNet_V1, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(11, 20),
            Swish(),
            nn.Linear(20, 20),
            Swish(),
            nn.Linear(20, 30),
            Swish(),
            nn.Linear(30, 30),
            Swish(),
            nn.Linear(30, 40),
            Swish(),
            nn.Linear(40, 40),
            Swish(),
            nn.Linear(40, 40),
            Swish(),
            nn.Linear(40, 20),
            Swish(),
            nn.Linear(20, 10),
            Swish(),
            nn.Linear(10, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.regressor(x).view(-1)
       
        return x


class SixLayerNet_V1(nn.Module):

    def __init__(self):
        super(SixLayerNet_V1, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(9, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x = self.regressor(x).view(-1)
       
        return x


class SixLayerNet_V2(nn.Module):

    def __init__(self):
        super(SixLayerNet_V2, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(11, 100),
            Swish(),
            nn.Linear(100, 200),
            Swish(),
            nn.Linear(200, 200),
            Swish(),
            nn.Linear(200, 200),
            Swish(),
            nn.Linear(200, 100),
            Swish(),
            nn.Linear(100, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.regressor(x).view(-1)
        
        return x


class SixLayerNet_V3(nn.Module):

    def __init__(self):
        super(SixLayerNet_V3, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(11, 50),
            Swish(),
            nn.Linear(50, 100),
            Swish(),
            nn.Linear(100, 100),
            Swish(),
            nn.Linear(100, 100),
            Swish(),
            nn.Linear(100, 50),
            Swish(),
            nn.Linear(50, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.regressor(x).view(-1)
        
        return x


class FourLayerNet_V1(nn.Module):

    def __init__(self):
        super(FourLayerNet_V1, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(11, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.regressor(x).view(-1)
       
        return x


class FourLayerNet_V2(nn.Module):

    def __init__(self):
        super(FourLayerNet_V2, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(11, 100),
            Swish(),
            nn.Linear(100, 100),
            Swish(),
            nn.Linear(100, 100),
            Swish(),
            nn.Linear(100, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.regressor(x).view(-1)
        
        return x


class FourLayerNet_V3(nn.Module):

    def __init__(self):
        super(FourLayerNet_V3, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(11, 50),
            Swish(),
            nn.Linear(50, 100),
            Swish(),
            nn.Linear(100, 50),
            Swish(),
            nn.Linear(50, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.regressor(x).view(-1)
        
        return x


class FourLayerNet_V4(nn.Module):

    def __init__(self):
        super(FourLayerNet_V4, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(7, 28),
            Swish(),
            nn.Linear(28, 14),
            Swish(),
            nn.Linear(14, 14),
            Swish(),
            nn.Linear(14, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.regressor(x).view(-1)
        
        return x


class FourLayerNet_V5(nn.Module):

    def __init__(self):
        super(FourLayerNet_V5, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(1, 4),
            Swish(),
            nn.Linear(4, 2),
            Swish(),
            nn.Linear(2, 2),
            Swish(),
            nn.Linear(2, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.regressor(x).view(-1)
        
        return x


class ThreeLayerNet_V2(nn.Module):

    def __init__(self):
        super(ThreeLayerNet_V2, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(11, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.regressor(x).view(-1)
        
        return x


class ThreeLayerNet_V3(nn.Module):

    def __init__(self, mode):
        super(ThreeLayerNet_V3, self).__init__()
        self.mode = mode
        self.regressor = nn.Sequential(
            nn.Linear(11, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.regressor(x).view(-1)
        
        return x


class ThreeLayerNet_V4(nn.Module):

    def __init__(self):
        super(ThreeLayerNet_V4, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(11, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.regressor(x).view(-1)

        return x


class TwoLayerNet_V2(nn.Module):

    def __init__(self):
        super(TwoLayerNet_V2, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(11, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.regressor(x).view(-1)
        
        return x


class SVR(nn.Module):

    def __init__(self):
        super(SVR, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(5, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.regressor(x).view(-1)
        
        return x

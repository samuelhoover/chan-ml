import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def pad_grids(grids):
    """
    Pad a sequence of nonuniformly-sized grids to a uniform size.

    Args:
        grids: sequence of grids

    Returns:
        padded_grids: sequence of padded grids
        orig_shapes: sequence of original grid sizes
    """

    # Find largest all-encompassing grid size
    orig_shapes = np.zeros((grids.shape[0], 3)).astype(int)
    for i, grid in enumerate(grids):
        orig_shapes[i, :] = grid.shape
    max_shape = np.max(orig_shapes, axis=0)

    # Pad each grid to size of largest grid that encompasses all grids
    # Save original size of each grid to be used later for reshaping
    template = np.zeros(max_shape)
    padded_grids = np.zeros((grids.shape[0], template.shape[0], template.shape[1], template.shape[2]))
    for idx, _ in enumerate(grids):
        size_diff = np.asarray(template.shape) - np.asarray(grids[idx].shape)
        padded_grids[idx, :, :, :] = (np.pad(grids[idx], [(0, size_diff[0]), (0, size_diff[1]), (0, size_diff[2])],
                                             mode='constant'))

    return padded_grids, orig_shapes


def unpad_grids(padded_grids, orig_shapes):
    """
    Unpad a sequence of padded uniformly-sized grids to their original, nonuniform size.

    Args:
        padded_grids: sequence of padded grids
        orig_shapes: sequence of original grid sizes

    Returns:
        orig_grids: sequence of grids at their original sizes
    """

    orig_grids = []  # need to store unpadded grids in a Python list
    for _, (padded_grid, orig_shape) in enumerate(zip(padded_grids, orig_shapes)):
        orig_grids.append(padded_grid[:orig_shape[0], :orig_shape[1], :orig_shape[2]])

    return orig_grids


def get_shortest_len(grids):
    """
    Find the shortest length in any dimension among a sequence of grids.

    Args:
        grids: sequence of grids

    Returns:
        min_size: shortest length in any dimension
    """

    min_size = np.inf
    for grid in grids:
        sizes = grid.shape
        if np.min(sizes) < min_size:
            min_size = np.min(sizes).astype(int)

    return min_size


# Unsure I will continue to use this format but leaving up for archival purposes
def train_val_test_split(X, y, train_size, val_size, test_size):
    """
    Split data into training, validation, and test sets. 
    ***train_size, val_size, and test_size must sum to 1.0.***

    Args:
        X: inputs
        y: targets
        train_size: proportion of data set to be included in training set
        val_size: proportion of data set to be included in validation set
        test_size: proportion of data set to be included in testing set

    Returns:
        X_train: training inputs
        y_train: training targets
        X_valid: validation inputs
        y_valid: validation targets
        X_test: testing inputs
        y_test: testing targets
    """

    assert np.sum([train_size, val_size, test_size]) == 1.0, "Train, valididation, and test split proportions must sum to 1."

    # need to compute intermediate split sizes from user-specified split sizes
    _train_size = 1 - test_size
    _val_size = 1 - (train_size / _train_size)

    if test_size == 0.0:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=_val_size, random_state=8)

        return (X_train, y_train), (X_valid, y_valid), (None, None)
    
    if val_size == 0.0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=14)

        return (X_train, y_train), (None, None), (X_test, y_test)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=12)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=_val_size, random_state=13)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def moving_average(x, window=5):
    """
    Calculate the moving average of an series.

    Args:
        x: data series
        window: size of moving window

    Returns:
        moving average
    """
    
    return np.convolve(x, np.ones(window), mode='valid') / window


def build_optimizer(net, args):
    """
    Construct optimizer.

    Args:
        net: initialized neural network
        args: optimizer arguments

    Returns:
        optimizer: initialized optimizer
    """

    if args.optim.lower() == 'sgd':
        optimizer = optim.SGD(
            params=net.parameters(),
            lr=args.init_lr,
            momentum=args.momentum,
            weight_decay=args.l2_reg,
            dampening=args.dampening,
        )
    
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(
            params=net.parameters(),
            lr=args.init_lr,
            betas=args.betas,
            eps=1e-8,
            weight_decay=args.l2_reg,
        )
    
    if args.optim.lower() == 'rmsprop':
        optimizer = optim.RMSProp(
            params=net.parameters(),
            lr=args.init_lr,
            momentum=args.momentum,
            alpha=args.alpha,
            eps=1e-8,
            centered=args.centered,
            weight_decay=args.l2_reg,
        )

    return optimizer


def build_lr_scheduler(optimizer, args):
    """
    Construct lr scheduler.

    Args:
        optimizer: initialized optimizer
        args: lr scheduler arguments

    Returns:
        scheduler: initialized lr scheduler
    """

    if args.lr_scheduler.lower() == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
            last_epoch=-1,
            verbose=False,
        )

    if args.lr_scheduler.lower() == 'reducelronplateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, 
            mode='min',
            factor=args.factor,
            patience=args.patience,
            threshold=args.threshold,
            threshold_mode='rel',
            cooldown=args.cooldown,
            min_lr=args.min_lr,
            eps=1e-8,
            verbose=True,
        )

    if args.lr_scheduler.lower() == 'cosineannealinglr':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, 
            T_max=args.t_max, 
            eta_min=args.min_lr, 
            last_epoch=-1, 
            verbose=False,
        )
    
    return scheduler


def build_loss(args):
    """
    Construct loss function.

    Args:
        args: loss arguments

    Returns:
        criterion: loss function
    """

    if args.loss_fn.lower() == 'mae':
        return nn.L1Loss()

    if args.loss_fn.lower() in ['smoothl1', 'huber']:
        return nn.SmoothL1Loss()
    
    if args.loss_fn.lower() == 'mse':
        return nn.MSELoss()

    if args.loss_fn.lower() == 'rmse':
        return RMSELoss()
    
    if args.loss_fn.lower() == 'rmsle':
        return RMSLELoss()

    if args.loss_fn.lower() == 'logcosh':
        return LogCosh_Loss()
    

class RMSELoss(nn.Module):
    """Root mean squared error loss."""

    def __init__(self, eps=1e-8):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred, true):
        return torch.sqrt(self.mse(pred, true) + self.eps)


class RMSLELoss(nn.Module):
    """Root mean squared log error loss."""

    def __init__(self, eps=1e-8):
        super(RMSLELoss, self).__init__()
        self.eps = eps

    def forward(self, pred, true):
        return torch.sqrt(torch.mean((torch.log(pred + 1) - torch.log(true + 1)) ** 2) + self.eps)


class LogCosh_Loss(nn.Module):
    """
    Log hyperbolic cosine loss. 
    Approximately equal to (x ^ 2) / 2 for small x and abs(x) - log(2) for large x.
    """

    def __init__(self):
        super(LogCosh_Loss, self).__init__()

    def forward(self, pred, true):
        return torch.mean(torch.log(torch.cosh(pred - true)))


class Swish(nn.Module):
    """Swish activation function."""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class LearnedSwish(nn.Module):
    """Swish activation function with learnable slope."""

    def __init__(self, slope=1.0):
        super(LearnedSwish, self).__init__()
        self.slope = slope

    def forward(self, x):
        return self.slope * x * torch.sigmoid(x)

import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from parser import parse_args
from run import run_train
from utils.data import load_data
from utils.vis_utils import plot_CV_results


# Deprecated but leaving up for archival purposes. Not implemented for use in any other scripts.
# Could be useful as a template for simple two variable hyperparameter tuning.

def validate(net, data, args):
    """
    Make predictions using a neural network.

    Args:
        net: trained neural network
        data: Dataset containing examples to make predictions on
        args: arguments
    Returns:
        MAE: mean absolute error
    """

    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    num_inf = len(dataloader)
    
    criterion = nn.L1Loss()
    running_loss = 0.0
    net.eval()
    for i, (X_batch, y_batch) in enumerate(dataloader):
        with torch.no_grad():
            batch_preds = net(X_batch)
            loss = criterion(batch_preds, y_batch)
            running_loss += loss.item()

    return running_loss / num_inf


def hpt(args):
    """
    Perform hyperparameter tuning.

    Args:
        args: arguments
    """
    
    LR = [5e-3, 1e-3, 5e-4, 1e-4]
    REG = [1e-5, 1e-3, 1e-1, 1]
    results = {}
    args = parse_args()
    train_data, valid_data = load_data(args)
    for lr in LR:
        args.init_lr = lr  # since using ArgParser, need to overwrite parsed args for specified hyperparameter
        for reg in REG:
            args.l2_reg = reg
            args.print_every = args.num_epochs + 1  # to suppress output
            net = run_train(net, args)
            t_loss = validate(net, data=train_data, args=args)
            v_loss = validate(net, data=valid_data, args=args)
            print('lr: {}, l2_reg: {}, train mae: {:.6f}, valid mae: {:.6f}'.format(lr, reg, t_loss, v_loss))
            results[(lr, reg)] = (t_loss, v_loss)

    fig = plot_CV_results(results)

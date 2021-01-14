import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from time import strftime

import networks.zeoliteDNN as dnn  # other networks to import, only import DNNs here

from parser import parse_args, write_config_log
from predict import predict
from train import train, train_valid
from utils.data import read_chan_data, read_grid_data, load_data


def run_train(net, time_stamp, args):
    """
    Train a neural network.

    Args:
        net: initialized neural network
        time_stamp: time stamp for documentation purposes
        args: training arguments

    Returns:
        net: trained neural network
    """

    if args.validate.lower() in ['true', 't', 'yes', 'y']:
        train_data, valid_data = load_data(args)
        net = train_valid(net, train_data=train_data, valid_data=valid_data, time_stamp=time_stamp, args=args)

    elif args.validate.lower() in ['false', 'f', 'no', 'n']:
        train_data = load_data(args)
        net = train(net, train_data=train_data, time_stamp=time_stamp, args=args)

    else:
        raise ValueError('Unrecognized validation argument -- options: [True/False, T/F, Y/N, Yes/No]')

    return net


def run_predict(net, time_stamp, args):
    """
    Make predictions using a neural network.

    Args:
        net: trained neural network
        time_stamp: time stamp for documentation purposes
        args: training arguments
    """

    # Currently set to use validation data to avoid fitting to test data.
    # Should only touch test data as the very last step for model evaluation i.e. don't validate model on test data.

    valid_data = load_data(args)
    predict(net, data=valid_data, time_stamp=time_stamp, args=args)


def run():
    """Run NN operation."""

    time_stamp = strftime('-%Y_%m_%d-%H_%M_%S')  # time stamp for documentation

    # Perhaps there would be a better way of passing an arg for what network to use but I had too many
    # networks I was testing to do something in the likes of `build_optimizer` or `build_loss`
    args = parse_args()
    if args.mode.lower() == 'train':
        net = dnn.ThreeLayerNet_V4()
        run_train(net, time_stamp, args)

    if args.mode.lower() == 'predict':
        net = dnn.FourLayerNet_V3()
        net.load_state_dict(torch.load(args.net_path))
        run_predict(net, time_stamp, args)

    write_config_log(net, time_stamp, args)

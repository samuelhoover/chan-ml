from argparse import ArgumentParser


def parse_args():
    """
    Parse arguments from text files.

    Text files:
        gen_args.txt - general arguments
        train_args.txt - training arguments
        predict_args.txt - prediction arguments

    Returns:
        args: parsed arguments
    """

    parser = ArgumentParser(fromfile_prefix_chars='@')  # main parser
    
    ### General main arguments (for gen_args.txt) ###
    parser.add_argument('--type', default='channel-based', choices=['channel-based', 'energy-based'], type=str,
                        help='Type [options: channel-based (default), energy-based]')
    parser.add_argument('--data_path', type=str,
                        help='Directory containing data')
    parser.add_argument('--mode', default='train', choices=['train', 'predict'], type=str,
                        help='Mode [options: train (default), predict]')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Mini-batch size [default: 32]')
    parser.add_argument('--train_size', default=0.6, type=float,
                        help='Proportion of the data to be set aside as train set [default: 0.6]')
    parser.add_argument('--val_size', default=0.2, type=float,
                        help='Proportion of the data to be set aside as validation set [default: 0.2]')
    parser.add_argument('--test_size', default=0.2, type=float,
                        help='Proportion of the data to be set aside as test set [default: 0.2]')

    subparsers = parser.add_subparsers(help='Add separate arguments for training or prediction')
    train_parser = subparsers.add_parser('train')  # subparser for training arguments
    predict_parser = subparsers.add_parser('predict')  # subparser for prediction arguments

    ### General training arguments (for train_args.txt) ###
    train_parser.add_argument('--num_epochs', default=50, type=int,
                              help='Number of epochs [default: 50]')
    train_parser.add_argument('--loss_fn', default='MSE', choices=['MAE', 'Huber', 'MSE', 'RMSE', 'RMSLE', 'LogCosh'], type=str,
                              help='Loss function [options: MAE, Huber, MSE (default), RMSE, RMSLE, LogCosh')
    train_parser.add_argument('--validate', default=False,
                              help='Perform validation [default: False]')
    train_parser.add_argument('--lr_scheduler', default='ReduceLROnPlateau', 
                              choices=['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'], type=str, 
                              help='lr scheduler [options: StepLR, ReduceLROnPlateau (default), CosineAnnealingLR]')
    ## General optimizer arguments ##
    train_parser.add_argument('--optim', default='SGD', choices=['SGD', 'Adam','RMSProp'], type=str,
                              help='Optimization algorithm [options: SGD (default), Adam, RMSProp]')
    train_parser.add_argument('--init_lr', default=1e-3, type=float,
                              help='Initial lr [default: 1e-3]')
    train_parser.add_argument('--l2_reg', default=0.0, type=float,
                              help='L2 regularization')
    train_parser.add_argument('--momentum', default=0.9,  type=float,
                              help='Momentum factor for SGD and RMSProp [default: 0.9]')
    # SGD-specific arguments #
    train_parser.add_argument('--dampening', default=0.0, type=float,
                              help='Dampening for momentum for SGD [default: 0.0]')
    train_parser.add_argument('--nesterov', default=False,
                              help='Enable Nesterov momentum for SGD [default: False]')
    # Adam-specific arguments #
    train_parser.add_argument('--betas', nargs=2, default=(0.9, 0.999), type=float,
                              help='Coefficients used for computing running averages of gradient and its square for Adam \
                              [default: (0.9, 0.999)]')
    # RMSProp-specific arguments #
    train_parser.add_argument('--alpha', default=0.99, type=float,
                              help='Smoothing constant for RMSProp [default: 0.99]')
    train_parser.add_argument('--centered', default=False,
                              help='Compute the centered RMSProp, the gradient is normalized by an estimation of its variance \
                              [default: False]')
    ## General learning rate scheduler arguments ##
    train_parser.add_argument('--factor', default=0.1, type=float,
                              help='Multiplicative factor to reduce lr for StepLR or ReduceLROnPlateau [default: 0.1]')
    train_parser.add_argument('--min_lr', default=0, type=float,
                              help='Minimum learning rate for StepLR or CosineAnnealingLR[default: 0]')
    # StepLR learning rate scheduler specific arguments #
    train_parser.add_argument('--step_size', default=10, type=int,
                              help='Number of epochs before reducing lr for StepLR [default: 10]')
    # ReduceLROnPlateau learning rate scheduler specific arguments #
    train_parser.add_argument('--patience', default=10, type=int,
                              help='Epochs outside threshold before reducing lr for ReduceLROnPlateau [default: 10]')
    train_parser.add_argument('--threshold', default=1e-4, type=float,
                              help='Threshold for measuring the new optimum for ReduceLROnPlateau [default: 1e-4]')
    train_parser.add_argument('--cooldown', default=0, type=int,
                              help='Epochs to wait to resume operation after lr has been reduced for ReduceLROnPlateau [default: 0]')
    # CosineAnnealingLR learning rate scheduler specific arguments # 
    train_parser.add_argument('--t_max', default=50, type=int,
                              help='Number of epochs until lr resets for CosineAnnealingLR [default: 50]')
    ## Gradient norm clipping arguments ##
    train_parser.add_argument('--max_norm', default=5, type=float,
                              help='Max norm of the gradients [default: 5]')
    ## Printing arguments ##
    train_parser.add_argument('--print_every', default=10, type=int,
                              help='Interval of epochs to print results [default: 10]')
    ## Post-training arguments ##
    train_parser.add_argument('--save_net', default=False,
                              help='Save network parameters [default: False]')
    train_parser.add_argument('--plot_loss', default=False,
                              help='Plot and save training curves [default: False]')

    ### General prediction arguments (for predict_args.txt) ###
    predict_parser.add_argument('--save_dir', type=str,
                                help='Directory to save predictions and targets to')
    predict_parser.add_argument('--net_path', type=str,
                                help='Path to load trained network from')

    return parser.parse_args()


def write_config_log(net, time_stamp, args):
    """
    Write a config log file of arguments.
    
    Args:
        net: neural network
        time_stamp: time stamp for documentation
        args: parsed training arguments
    """

    # Clean up training arguments before printing to config file
    if args.mode.lower() == 'train':
        logdir = 'configs/train'
        
        # Delete non-SGD-specific arguments if using SGD
        if args.optim.lower() == 'sgd':
            del args.betas
            del args.alpha
            del args.centered

        # Delete non-Adam-specific arguments if using Adam
        if args.optim.lower() == 'adam':
            del args.momentum
            del args.dampening
            del args.nesterov
            del args.alpha
            del args.centered
        
        # Delete non-RMSProp-specific arguments if using RMSProp
        if args.optim.lower() == 'rmsprop':
            del args.betas
            del args.dampening
            del args.nesterov

        # Delete non-StepLR-specific arguments in using StepLR
        if args.lr_scheduler.lower() == 'steplr':
            del args.patience
            del args.threshold
            del args.cooldown
            del args.min_lr
            del args.t_max

        # Delete non-ReduceLROnPlateau-specific arguments if using ReduceLROnPlateau
        if args.lr_scheduler.lower() == 'reducelronplateau':
            del args.step_size
            del args.t_max

        # Delete non-CosineAnnealingLR-specific arguments if using CosineAnnealingLR
        if args.lr_scheduler.lower() == 'cosineannealinglr':
            del args.step_size
            del args.factor
            del args.patience
            del args.threshold
            del args.cooldown

        del args.print_every
        del args.save_net
        del args.plot_loss
    
    # Clean up prediction arguments before printing to config file
    if args.mode.lower() == 'predict':
        logdir = 'configs/predict'

        del args.save_dir

    # Write cleaned up arguments to config file
    arg_dict = list(args.__dict__.items())
    with open('{}/config{}'.format(logdir, time_stamp), mode='w') as f:
        f.write('{}\n'.format(net.__class__.__name__))
        for arg, val in arg_dict:
            f.write('{} {}\n'.format(arg, val))

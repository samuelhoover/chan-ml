import torch

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import trange

from utils.ops import build_optimizer, build_lr_scheduler, build_loss
from utils.vis_utils import plot_loss


def train(net, train_data, time_stamp, args):
    """
    Mini-batch train NN on provided data set.

    Args:
        net: initialized neural network
        train_data: dataset containing training inputs and targets
        time_stamp: time stamp for documentation purposes
        args: arguments      
    Returns:
        net: trained neural network   
    """

    # Load data
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Construct loss function, optimization method, and learning rate scheduler
    criterion = build_loss(args)
    optimizer = build_optimizer(net, args)
    scheduler = build_lr_scheduler(optimizer, args)

    # Initialize important parameters
    num_train = len(trainloader)
    num_epochs = args.num_epochs
    max_norm = args.max_norm
    train_loss = np.zeros(num_epochs)

    # Begin training
    for epoch in trange(num_epochs):
        running_loss = 0.0

        # Mini-batch training step
        net.train()
        for X_train_batch, y_train_batch in enumerate(trainloader):
            for param in net.parameters():  # clear out gradients from previous optimization step
                param.grad = None
            
            outputs = net(X_train_batch)
            loss = criterion(outputs, y_train_batch)
            loss.backward()  # calculate gradients
            nn.utils.clip_grad_norm_(net.parameters(), max_norm)  # clip gradient norm
            optimizer.step()  # update weights
            running_loss += loss.item()
        
        # Print real-time results
        if (epoch + 1) % args.print_every == 0:
            print('Epoch: {:>6d}/{:d},    {}: {:.10f}'.format(
                epoch + 1, num_epochs, criterion.__class__.__name__, running_loss / num_train,
                ),
            )
        
        train_loss[epoch] = running_loss / num_train

        # Update learning rate if necessary
        scheduler.step(loss)

    # Plot learning curve if specified    
    if args.plot_loss.lower() in ['true', 't', 'yes', 'y']:
        figdir = ''.join(['figures/training-curves/', net.__class__.__name__ + time_stamp, '.png'])
        fig = plot_loss(
            train_loss=train_loss.flatten(), 
            title=net.__class__.__name__, 
            loss_fn=criterion.__class__.__name__,
        )
        plt.show()
        fig.savefig(figdir, dpi=300, bbox_inches='tight')

    # Save trained network if specified
    if args.save_net.lower() in ['true', 't', 'yes', 'y']:
        print('Saving the trained model ...')
        save_path = ''.join(['saved-networks/', net.__class__.__name__ + time_stamp, '.pth'])
        torch.save(net.state_dict(), save_path)
        print('Saved model')

    return net


def train_valid(net, train_data, valid_data, time_stamp, args):
    """
    Mini-batch train and validate NN on provided data set.

    Args:
        net: neural network
        train_data: data set containing training inputs and targets
        valid_data: data set containing validation inputs and targets
        time_stamp: time stamp for documentation purposes
        args: arguments
    """
    
    # Load data
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    validloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Construct loss function, optimization method, and learning rate scheduler
    criterion = build_loss(args)
    optimizer = build_optimizer(net, args)
    scheduler = build_lr_scheduler(optimizer, args)

    # Initialize important parameters
    num_train = len(trainloader)
    num_valid = len(validloader)
    num_epochs = args.num_epochs
    max_norm = args.max_norm
    train_loss = np.zeros(num_epochs)
    valid_loss = np.zeros(num_epochs)

    # Begin training
    for epoch in trange(num_epochs):
        t_loss = 0.0
        v_loss = 0.0

        # Mini-batch training step
        net.train()
        for X_train_batch, y_train_batch in trainloader:
            for param in net.parameters():  # clear out gradients from previous optimization step
                param.grad = None
            
            outputs = net(X_train_batch)  # make predictions
            loss = criterion(outputs, y_train_batch)  # calculate loss
            loss.backward()  # calculate gradients
            nn.utils.clip_grad_norm_(net.parameters(), max_norm)  # clip gradient norm
            optimizer.step()  # update weights
            t_loss += loss.item()
        
        # Validation step
        net.eval()
        with torch.no_grad():
            for X_valid_batch, y_valid_batch in validloader:
                outputs = net(X_valid_batch)
                loss = criterion(outputs, y_valid_batch)
                v_loss += loss.item()
    
        # Print real-time results
        if (epoch + 1) % args.print_every == 0:
            print('Epoch: {:>6d}/{:d},    Train loss: {:.10f},    Valid loss: {:.10f}'.format(
                epoch + 1, num_epochs, t_loss / num_train, v_loss / num_valid,
                ),
            )
    
        train_loss[epoch] = t_loss / num_train
        valid_loss[epoch] = v_loss / num_valid

        # Update learning rate if necessary
        scheduler.step(v_loss)  # LR scheduler step can be dependent upon either training or validation loss
        
    # Plot learning curves if specified
    if args.plot_loss.lower() in ['true', 't', 'yes', 'y']:
        figdir = ''.join(['figures/training-curves/', net.__class__.__name__ + time_stamp, '.png'])
        fig = plot_loss(
            train_loss=train_loss.flatten(), 
            valid_loss=valid_loss.flatten(),
            title=net.__class__.__name__, 
            loss_fn=criterion.__class__.__name__,
        )
        plt.show()
        fig.savefig(figdir, dpi=300, bbox_inches='tight')

    # Save trained network if specified
    if args.save_net.lower() in ['true', 't', 'yes', 'y']:
        print('Saving the trained model ...')
        save_path = ''.join(['saved-networks/', net.__class__.__name__ + time_stamp, '.pth'])
        torch.save(net.state_dict(), save_path)
        print('Saved model')

    return net

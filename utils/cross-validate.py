import torch

import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import TensorDataset, DataLoader

from utils.data import load_data
from utils.ops import build_loss, build_optimizer, build_lr_scheduler, RMSELoss


def kfold_train(net, train_data, args):
    """
    Training for k-fold cross-validation.

    Args:
        net: initialized neural network
        train_data: Dataset containing training examples
        args: arguments      
    Returns:
        net: trained neural network 
        train_loss: training loss  
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
    train_loss = 0.0

    for epoch in range(num_epochs):
        net.train()
        for X_train_batch, y_train_batch in trainloader:
            for param in net.parameters():  # clear out gradients from previous optimization step
                param.grad = None
            
            outputs = net(X_train_batch)
            loss = criterion(outputs, y_train_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step(loss)
    
    train_loss /= num_train

    return net, train_loss


def kfold_validate(net, valid_data, args):
    """
    Validation for k-fold cross-validation.

    Args:
        net: trained neural network
        valid_data: Dataset containing validation examples
        args: arguments
    Returns:
        valid_loss: validation loss
    """

    # Load data
    dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    # Build loss
    criterion = build_loss(args)

    # Initialize important parameters
    num_val = len(dataloader)
    valid_loss = 0.0

    net.eval()
    for X_batch, y_batch in dataloader:
        with torch.no_grad():
            batch_preds = net(X_batch)
            loss = criterion(batch_preds, y_batch)
            valid_loss += loss.item()

    valid_loss /= num_val

    return valid_loss


def kfold_step(net, train_data, valid_data, args):
    """
    Perform a k-fold step.

    Args:
        net: initialized neural network
        train_data: PyTorch Dataset for training data
        valid_data: PyTorch Dataset for validation data
        args: arguments

    Returns:
        net: trained neural network
        train_loss: fold training loss
        valid_loss fold validation loss
    """

    net, train_loss = kfold_train(net, train_data=train_data, args=args)
    valid_loss = kfold_validate(net, valid_data=valid_data, args=args) 

    return net, train_loss, valid_loss


def kfold_CV(net, args, k=5):
    """
    Perform k-fold cross-validation.

    Args:
        net: initialized neural network
        args: arguments
        k: number of folds for k-fold cross-validation [default: 5]
    """

    # Load data
    df = pd.read_csv(args.data_path)
    X = df.drop('kH_C18').values
    y = df['kH_C18'].values
    if y.max() < 1e7: 
         y *= 1e6  # convert mol/kg/Pa to mol/kg/MPa

    # Split data into train and test sets
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

    # Scale features
    transformer = RobustScaler().fit(X_train)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)

    # Parameters for finding "best" network
    best_score = np.inf  # arbitrarily large number
    best_net = None

    kf = KFold(n_splits=k, shuffle=False, random_state=3)
    for train_index, valid_index in kf.split(X_train):
        train_data = TensorDataset(torch.Tensor(X_train[train_index]), torch.Tensor(y_train[train_index]))
        valid_data = TensorDataset(torch.Tensor(X_train[valid_index]), torch.Tensor(y_train[valid_index]))

        net, train_loss, valid_loss = kfold_step(net, train_data, valid_data, args)
        print('Train loss: {:.6f},  Validation loss: {:.6f}'.format(train_loss, valid_loss))

        if valid_loss < best_score:
            best_score = valid_loss
            best_net = net

    print('Saving the best model ...')
    torch.save(best_net.state_dict(), 'saved-networks/cv/zeo' + best_net.__class__.__name__ + '.pth')
    print('Saved model')

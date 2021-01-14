import os
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import trange
from tqdm.contrib import tenumerate

from utils.ops import RMSELoss
from utils.vis_utils import plot_results_dist, plot_error_dist, parity_plot


def predict(net, data, time_stamp, args):
    """
    Make predictions using a neural network.

    Args:
        net: trained neural network
        data: Dataset containing examples to make predictions on
        args: arguments
    """

    # Load data
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Construct loss function
    criterion = RMSELoss()

    # Initialize important
    num_pred = len(dataloader) 
    preds = np.zeros(num_pred * args.batch_size)
    targets = np.zeros(num_pred * args.batch_size)
    running_loss = 0.0

    net.eval()
    for i, batch in tenumerate(dataloader):
        X_batch, y_batch = batch
        with torch.no_grad():
            batch_preds = net(X_batch)
            loss = criterion(batch_preds, y_batch)
            running_loss += loss.item()
        
        preds[i*args.batch_size:(i+1)*args.batch_size] = batch_preds.detach().numpy()
        targets[i*args.batch_size:(i+1)*args.batch_size] = y_batch.detach().numpy()

    print('\nRMSE: {:.6f}'.format(running_loss / num_pred))
    
    results = pd.DataFrame(np.hstack((preds.reshape(-1, 1), targets.reshape(-1, 1))), columns=['predictions', 'targets'])
    print('==================================\n')
    print(results.describe())
    print('\n==================================\n')
    
    # Save predictions and targets
    save_path = os.path.join(args.save_dir, net.__class__.__name__ + time_stamp + '.csv')
    results.to_csv(save_path)

    # Save prediction and targets distributions figure
    dist_figdir = ''.join(['figures/prediction-target-distributions/', net.__class__.__name__, time_stamp, '.png'])
    fig = plot_results_dist(results=results)
    fig.savefig(dist_figdir, dpi=300, bbox_inches='tight')
    plt.close()

    # Save absolute error distribution figure
    error_figdir = ''.join(['figures/prediction-error-distributions/', net.__class__.__name__, time_stamp, '.png'])
    fig = plot_error_dist(results=results)
    fig.savefig(error_figdir, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save parity plot figure
    parity_figdir = ''.join(['figures/parity-plots/', net.__class__.__name__, time_stamp, '.png'])
    fig = parity_plot(results=results)
    fig.savefig(parity_figdir, dpi=300, bbox_inches='tight')
    
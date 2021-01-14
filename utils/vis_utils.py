import torch

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch.nn as nn

from sklearn.metrics import mean_squared_error, r2_score


def plot_loss(train_loss, valid_loss=None, title=None, loss_fn=None):
    """
    Plot training curve. Optionally, plot validation curve.

    Args:
        train_loss: array of each epoch's training loss
        valid_loss (optional): array of each epoch's validation loss
        title (optional): figure title
        loss_fn (optional): loss function used
    """

    num_epochs = len(train_loss)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(np.arange(1, num_epochs + 1), train_loss, label='Train')
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(loss_fn)

    if valid_loss is not None:
        ax.plot(np.arange(1, num_epochs + 1), valid_loss, label='Validation')

    if train_loss.max() >= 1e6:
        ax.set_yscale('log')

    ax.legend()
    ax.grid(b=True, which='both', axis='both', color='grey', alpha=0.2)
    fig.tight_layout()

    return fig


def volshow(volume, title):
    """
    Visualize NumPy or PyTorch volume data.

    Args:
        volume: NumPy or PyTorch volume data
        title: figure title
    """

    if isinstance(volume, np.ndarray):
        a, b, c = np.squeeze(volume).shape
    
    else:
        a, b, c = torch.squeeze(volume).size()

    x, y, z = np.mgrid[0:a, 0:b, 0:c]
    
    fig = go.Figure(
        data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=volume.flatten(),
            isomin=0.1,
            isomax=0.8,
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=25,  # needs to be a large number for good volume rendering
        ),
    )

    fig.update_layout(
        title='{}'.format(title),
        scene=dict(
            xaxis_title='a',
            yaxis_title='b',
            zaxis_title='c',
        ),
    )

    fig.show()

    return fig


def contourfshow(volume, zeolite, ax=None, slice_dir='x', slice_idx=None):
    """
    Generate filled contour of a cross-section of energy grid volume data.

    Args:
        volume: energy grid volume data
        zeolite: zeolite structure name
        slice_axis (optional): slice axis [options: 'x' (default), 'y', or 'z']
        slice_idx (optional): slice index
    """

    if not isinstance(volume, np.ndarray):
        volume = np.array(volume)
    
    # if AdaptiveMaxPool3D layer was used, it unsqueezes volume (3D) data into 4D data
    volume = np.squeeze(volume)

    if slice_idx is None:
        if slice_dir.lower() == 'x':
            slice_idx = np.random.randint(0, volume.shape[0])

        if slice_dir.lower() == 'y':
            slice_idx = np.random.randint(0, volume.shape[1])

        if slice_dir.lower() == 'z':
            slice_idx = np.random.randint(0, volume.shape[2])    

    if slice_dir.lower() == 'x':
        slice = np.squeeze(volume[slice_idx, :, :])
        xlabel = 'y'
        ylabel = 'z'

    elif slice_dir.lower() == 'y':
        slice = np.squeeze(volume[:, slice_idx, :])
        xlabel = 'x'
        ylabel = 'z'

    else:
        slice = np.squeeze(volume[:, :, slice_idx])
        xlabel = 'x'
        ylabel = 'y'

    a, b = np.meshgrid(
        np.linspace(0, slice.shape[0] - 1, slice.shape[0]),
        np.linspace(0, slice.shape[1] - 1, slice.shape[1]),
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.contourf(a, b, np.transpose(slice))  # need to tranpose the slice, not sure why
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('{} at {} = {:.1f} Å'.format(zeolite, slice_dir, slice_idx * 0.2))

    fig.colorbar()

    return fig


def plot_error_dist(results):
    """
    Generate distribution of prediction absolute errors.

    Args:
        results: Pandas DataFrame containing targets and predictions
    """

    abs_error = abs(results.predictions - results.targets)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(np.log1p(abs_error), bins=100, edgecolor='black')
    ax.set_yscale('log')
    ax.set_xlabel('log$_{10}(1 + error)$')
    ax.set_ylabel('Frequency')

    fig.tight_layout()
    plt.show()

    return fig


def plot_results_dist(results):
    """
    Generate distributions for targets and predictions.

    Args:
        results: Pandas DataFrame containing targets and predictions
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    axes[0].set_title('Predicted')
    axes[0].hist(results.predictions, bins=100, edgecolor='black')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('$k_{{H,C_{{18}}}}$ [mol/kg/MPa]')
    axes[0].set_ylabel('Frequency')

    axes[1].set_title('True')
    axes[1].hist(results.targets, bins=100, edgecolor='black')
    axes[1].set_xlabel('$k_{{H,C_{{18}}}}$ [mol/kg/MPa]')
    axes[1].set_yscale('log')

    plt.suptitle('RMSE: {:.2f}'.format(np.sqrt(mean_squared_error(results.targets, results.predictions))))
    fig.tight_layout()
    plt.show()

    return fig


# adding a histogram on axes to show distributions of targets and predictions would be useful
def parity_plot(results, error_per=5):
    """
    Generate parity plot.

    Args:
        results: Pandas DataFrame containing targets and predictions
        error_per: error percentage to be highlighted
    """

    # error range to be highlighted
    x = np.linspace(0, results.max().max(), 2)  # one-to-one line
    plus_error = (1 + error_per / 100) * x
    minus_error = (1 - error_per / 100) * x

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    axes[0].set_title('full set')
    axes[0].scatter(results.targets, results.predictions, s=1)
    axes[0].plot([0, results.max().max()], [0, results.max().max()], '-k', linewidth=0.5)
    axes[0].fill_between(x, plus_error, minus_error, alpha=0.3, label='$\pm$5% error')
    axes[0].set_xlabel('true')
    axes[0].set_ylabel('predicted')
    axes[0].axis([0, results.max().max(), 0, results.max().max()])
    axes[0].text(
        results.max().max() / 10, results.max().max() * 0.9, '$R^2$={:.2f}, RMSE={:.2f}'.format(
            r2_score(results.targets, results.predictions),
            np.sqrt(mean_squared_error(results.targets, results.predictions)),
        ),
    )

    axes[1].set_title('zoomed in')
    axes[1].scatter(results.targets, results.predictions, s=1)
    axes[1].plot([0, 1e5], [0, 1e5], '-k', linewidth=0.5)
    axes[1].fill_between(x, plus_error, minus_error, alpha=0.3)
    axes[1].set_xlabel('true')
    axes[1].set_ylabel('predicted')
    axes[1].axis([0, 1e5, 0, 1e5])
    
    fig.legend()
    fig.tight_layout()
    plt.show()

    return fig


# Deprecated but leaving up for archival purposes
def plot_HPT_results(results):
    """
    Plot 2 parameter hyperparameter tuning results.

    Args:
        results: hyperparameter tuning results
    """

    x_scatter = [x[0] for x in results]
    y_scatter = [x[1] for x in results]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    for i, ax in enumerate(axes.ravel()):
        colors = [results[x][i] for x in results]
        im = ax.scatter(x_scatter, y_scatter, 100, c=colors, cmap='jet')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('learning rate')
        ax.set_ylabel('regularization strength')
        if i == 0:
            ax.set_title('training MAE')

        if i ==1:
            ax.set_title('validation MAE')

    fig.tight_layout()
    plt.show()

    return fig

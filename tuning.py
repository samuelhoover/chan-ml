import torch

from ax.plot.contour import plot_contour
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from torchvision import transforms
from torch.utils.data import DataLoader

from networks.zeoliteAdaDNN import AdaThreeLayerNet
from networks.zeoliteCNN import TinyConvNet
from train import train
from utils.data import ZeoStructDataset, PeriodicPadding, ToTensor


# Perform hyperparameter tuning on PyTorch models using FACEBOOK's Ax
# https://ax.dev/  <-- for further details on Ax


def evaluate(net, data_loader):
    """
    Compute classification accuracy on provided dataset.

    Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
    Returns:
        float: MSE
    """

    net.eval()
    loss = 0
    total = 0
    with torch.no_grad():
        for sample in data_loader:
            inputs, target = sample
            target = target.view(-1, 1)  # reshaping to match input size
            outputs = net(inputs)
            loss += (target - outputs)**2
            total += 1

    return {'MSE': (loss / total)}


# When I first tried Ax, I was trying to learn from the energy grid files
def train_evaluate(parameterization):
    torch.manual_seed(12345)
    net = TinyConvNet(output_size=(24, 24, 24))
    train_set = ZeoStructDataset(
        csv_file='/Volumes/SH External HDD/Data/Zeolites/batch_V2/train/train_kH.csv',
        root_dir='/Volumes/SH External HDD/Data/Zeolites/batch_V2/train/',
        transform=transforms.Compose([PeriodicPadding(), ToTensor()]),
    )
    val_set = ZeoStructDataset(
        csv_file='/Volumes/SH External HDD/Data/Zeolites/batch_V2/val/val_kH.csv',
        root_dir='/Volumes/SH External HDD/Data/Zeolites/batch_V2/val/',
        transform=transforms.Compose([PeriodicPadding(), ToTensor()]),
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True)
    post_train = {
        'plot_loss': False,
        'plot_weights': False,
        'save_net': False,
    }
    net = train(net, train_data=train_set, batch_size=1, parameters=parameterization, post_train=post_train)
    return evaluate(net=net, data_loader=val_loader)


parameters = [
    {
        'name': 'lr',
        'type': 'range',
        'bounds': [1e-6, 0.4],
        'log_scale': True,
    },
    {
        'name': 'weight_decay',
        'type': 'range',
        'bounds': [0.0, 1.0],
    }
]
best_parameters, values, experiment, model = optimize(
    parameters=parameters,
    experiment_name='TinyConvNet tuning',
    objective_name='minimization',
    evaluation_function=train_evaluate,
    minimize=True,
)
print(best_parameters)
means, covariances = values
print(means, covariances)
render(plot_contour(model=model, param_x='lr', param_y='weight_decay', metric_name='MSE'))

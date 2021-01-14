import os
import torch

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms

from utils.ops import train_val_test_split


# Deprecated but leaving up for archival purposes. 
def read_chan_data_DEPRECATED(args):
    """
    Read in and preprocess zeolite chemistry-channel data.

    Args:
        args: arguments

    Returns:
        train_data: TensorDataset of training data
        valid_data: TensorDataset of validation data
        test_data: TensorDataset of testing data
    """

    if args.data_path.endswith('.csv'):  # if loading whole dataset from .csv file
        df = pd.read_csv(args.data_path)
        X, y = df[df.columns.drop('kH_C18')], df['kH_C18']

        # if kH_C18 in mol/kg/Pa units, convert mol/kg/Pa to mol/kg/MPa
        if y.max() <= 1e6:
            y *= 1e6

        if args.loss_fn.lower() in ['mae', 'huber', 'mse', 'rmse']:
            y = np.log1p(y)  # compress kH_C18 to log scale [np.log1p(y) == log(1 + y)]

        if args.val_size == 0.0:
            return

        # train/val/test split
        train, valid, test = train_val_test_split(
            X.values, y.values, 
            train_size=args.train_size, 
            val_size=args.val_size, 
            test_size=args.test_size,
        )

        # unpack sets into individual features and target sets
        X_train, y_train = train
        X_valid, y_valid = valid
        X_test, y_test = test
    
    else:  # if loading pre-split data from directory
        df_train = pd.read_csv(os.path.join(args.data_path, 'train.csv'), index_col=0)
        df_valid = pd.read_csv(os.path.join(args.data_path, 'valid.csv'), index_col=0)
        df_test = pd.read_csv(os.path.join(args.data_path, 'test.csv'), index_col=0)

        # unpack sets into individual target and features sets
        y_train, y_valid, y_test = df_train.pop('kH_C18').values, df_valid.pop('kH_C18').values, df_test.pop('kH_C18').values
        X_train, X_valid, X_test = df_train.values, df_valid.values, df_test.values

        # if kH_C18 in mol/kg/Pa units, convert mol/kg/Pa to mol/kg/MPa
        if np.array([y_train.max(), y_valid.max(), y_test.max()]).max() <= 1e6:
            y_train *= 1e6
            y_valid *= 1e6
            y_test *= 1e6

    # scale features
    X_transformer = RobustScaler().fit(X_train)
    X_train = X_transformer.transform(X_train)
    X_valid = X_transformer.transform(X_valid)
    X_test = X_transformer.transform(X_test)

    # pack split sets into TensorDatasets
    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    valid_data = TensorDataset(torch.Tensor(X_valid), torch.Tensor(y_valid))
    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    return train_data, valid_data, test_data


def read_chan_data(args):
    """
    Read in and preprocess zeolite chemistry-channel data.

    Args:
        args: arguments

    Returns:
        train_data: TensorDataset of training data
        valid_data: TensorDataset of validation data
        test_data: TensorDataset of testing data
    """

    # Read in data
    df_train = pd.read_csv(os.path.join(args.data_path, 'train.csv'), index_col=0)
    df_valid = pd.read_csv(os.path.join(args.data_path, 'valid.csv'), index_col=0)
    df_test = pd.read_csv(os.path.join(args.data_path, 'test.csv'), index_col=0)

    # Unpack sets into individual target and features sets
    X_train, y_train = df_train[df_train.columns.drop('kH_C18')].values, df_train['kH_C18'].values
    X_valid, y_valid = df_valid[df_valid.columns.drop('kH_C18')].values, df_valid['kH_C18'].values
    X_test, y_test = df_test[df_test.columns.drop('kH_C18')].values, df_test['kH_C18'].values

    # Convert kH_C18 from mol/kg/Pa to mol/kg/MPa, if needed
    if np.array([y_train.max(), y_valid.max(), y_test.max()]).max() <= 1e6:
        y_train *= 1e6
        y_valid *= 1e6
        y_test *= 1e6

    # Scale and normalize features
    X_transformer = RobustScaler().fit(X_train)
    X_train = X_transformer.transform(X_train)
    X_valid = X_transformer.transform(X_valid)
    X_test = X_transformer.transform(X_test)

    # Pack split sets into TensorDatasets
    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    valid_data = TensorDataset(torch.Tensor(X_valid), torch.Tensor(y_valid))
    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    return train_data, valid_data, test_data


def read_grid_data(args):
    """
    Read in and preprocess energy grid data.

    Args:
        args: arguments

    Returns:
        train_data: PyTorch Dataset of training data
        valid_data: PyTorch Dataset of validation data
        test_data: PyTorch Dataset of testing data
    """

    train_data = ZeoStructDataset(
        csv_file=os.path.join(args.data_path, 'train/train_kH.csv'),
        root_dir=os.path.join(args.data_path, 'train/'),
        transform=transforms.Compose([PeriodicPadding(), ToTensor()]),
    )
    
    if args.validate.lower() in ['true', 't', 'y', 'yes']:
        valid_data = ZeoStructDataset(
            csv_file=os.path.join(args.data_path, 'val/val_kH.csv'),
            root_dir=os.path.join(args.data_path, 'val/'),
            transform=transforms.Compose([PeriodicPadding(), ToTensor()]),
        )
    
    else:
        valid_data = None                                  

    test_data = ZeoStructDataset(
        csv_file=os.path.join(args.data_path, 'test/test_kH.csv'),
        root_dir=os.path.join(args.data_path, 'test/'),
        transform=transforms.Compose([PeriodicPadding(), ToTensor()]),
    )

    return train_data, valid_data, test_data


def load_data(args):
    """
    Load data for training, validation, or testing purposes.

    Args:
        args: arguments

    Returns:
        train_data: TensorDataset of training data
        valid_data: TensorDataset of validation data
        test_data: TensorDataset of tresting data
    """

    if args.type.lower() == 'channel-based':
        train_data, valid_data, test_data = read_chan_data(args)
        if args.mode.lower() == 'train':
            if args.validate.lower() in ['true', 't', 'yes', 'y']:
                return train_data, valid_data
            
            return train_data

        if args.mode.lower() == 'predict':  # still using validation data just to avoid touching testing data
            return valid_data  # can swap out for `test_data` if wanted

    if args.type.lower() == 'energy-based':
        train_data, valid_data, test_data = read_grid_data(args)
        if args.mode.lower() == 'train':
            if args.validate.lower() in ['true', 't', 'yes', 'y']:
                return train_data, valid_data
            
            return train_data

        if args.mode.lower() == 'predict':  # still using validation data just to avoid touching testing data
            return valid_data  # can swap out for `test_data` if wanted


class ZeoStructDataset(Dataset):
    """Zeolite structures dataset."""

    def __init__(self, csv_file, root_dir, transform=None):  # initialize class ??? [is this correct?]
        """
        Args:
            csv_file: path to the csv file with targets
            root_dir: path to the directory containing energy grids
            transform (optional): transform(s) to be applied on a sample

        Returns:
            sample: a tuple containing a zeolite energy grid and target
        """

        self.targets = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):  # get length/number of samples in dataset
        return len(self.targets)

    def __getitem__(self, idx):  # fetch sample from dataset
        if torch.is_tensor(idx):
            idx = idx.tolist()

        zeo_file = os.path.join(self.root_dir, self.targets.iloc[idx, 0] + '.npy')
        energy_grid = np.load(zeo_file, allow_pickle=True)
        targets = np.array(self.targets.iloc[idx, 1]).astype('float')
        sample = (energy_grid, targets)

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class PeriodicPadding(object):
    """Apply periodic boundary conditions to energy grids."""

    def __call__(self, sample):
        """
        Args:
            sample: tuple containing zeolite energy grid and target

        Returns:
            sample: tuple containing periodic padded zeolite energy grid and target
        """

        energy_grid, target = sample

        h, w, d = energy_grid.shape[:3]
        pad_h = h // 2
        pad_w = w // 2
        pad_d = d // 2

        padded_grid = np.pad(energy_grid, ((pad_h, pad_h), (pad_w, pad_w), (pad_d, pad_d)), mode='wrap')

        return (padded_grid, target)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """
        Args:
            sample: tuple containing zeolite energy grid and target
        
        Returns:
            sample: tuple containing PyTorch Tensors of zeolite energy grid and target
        """

        energy_grid, target = sample

        return (torch.from_numpy(energy_grid).float(), torch.from_numpy(target).float())

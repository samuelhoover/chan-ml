import os
import sys
import torch
import scipy.io

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.vis_utils import contourfshow
from utils.data import pad_grids


# RESULTS:
#   1.) PyTorch Tensors do not accept NumPy object arrays as inputs (MATLAB cell arrays are converted to NumPy object
#       array when importing through scipy.io.loadmat).
#   2.) Inputs to PyTorch Tensors must be uniform in shape (can't convert NumPy object array to Python list or Pandas
#       DataFrame then convert to PyTorch Tensor. Data in Pandas DataFrame are still it's inputs i.e. NumPy object).
#   3.) nn.AdaptiveMaxPool#D layer acts as a downsampling operation. Takes any size input are outputs a fixed sized
#       tensor.
#   4.) Lists of fields (zeolite structure names, energy grids, Henry's constants) are NumPy object arrays. Individual
#       elements are their respective data types i.e. zeolite structure names are strings and energy grids and Henry's
#       constants are doubles.
#   5.) Saving data to compressed format (.npz) compresses file about 6 times smaller than original size


# ???
def load_inspect_mat(mat_file):
    mat_struct = scipy.io.loadmat(mat_file, squeeze_me=True)
    print('mat_struct keys: {}\n'.format(mat_struct.keys()))
    struct_key = list(mat_struct.keys())[-1]  # MATLAB data structure is the last key when using scipy.io.loadmat
    struct = mat_struct[struct_key]
    print('struct shape: {}'.format(struct.shape))
    print('struct dtype: {}'.format(struct.dtype))
    print(struct[0])
    print('')

    return struct


def dataset_split_check(dataset):
    # Can I extract inputs and targets from TensorDataset?
    try:
        inputs, targets = dataset
    except ValueError:
        print("Can't split TensorDataset into inputs and targets tensors.")
    else:
        print("Can split TensorDataset into inputs and targets tensor.")
    # No, you can't split TensorDataset into inputs and targets tensors

    # Can I get a single input and target from TensorDataset?
    try:
        single_input, single_target = dataset[0]
    except ValueError:
        print("Can't split TensorDataset into single input and target.")
    else:
        print("Can split TensorDataset into single input and target.")
    # Yes, you can extract a single input and target from TensorDataset


def nested_list_check(nested_list):
    # Inspect nested list and contents
    print('Size of nested list: {} bytes'.format(sys.getsizeof(nested_list)))
    print('nested_list type: {}    nested_list length: {}'.format((type(nested_list), len(nested_list))))
    print('Firste example in the list type: {}    First example in list shape: {}\n'.format(
          (type(nested_list[0]), nested_list[0].shape)),
    )

    # Try converting nested Python list with nonuniform-sized elements to a PyTorch tensor
    try:
        tensor = torch.tensor(nested_list)
    except ValueError as e:
        print('ValueError: {}'.format(e))
        print('Can''t convert Python list to PyTorch Tensor if elements are not a uniform size.')
    else:
        print('Converted Python list with varying element sizes to PyTorch Tensor.')
    # No, you can't convert a Python list with nonuniform-sized elements to a PyTorch tensor


def amp3d_test(grid, zeo, kH):
    m = nn.AdaptiveMaxPool3d((25, 28, 31))
    pre_grid = torch.unsqueeze(torch.from_numpy(grid), 0)
    print('Before: {}'.format(pre_grid.size()))
    post_grid = m(pre_grid)
    print('After: {}\n'.format(post_grid.size()))
    plt.suptitle('{} [kH = {:.0f} mol/kg/MPa]'.format(zeo, kH))
    plt.subplot(1, 2, 1)
    contourfshow(grid, 'Original', slice_idx=1)
    plt.subplot(1, 2, 2)
    contourfshow(post_grid, 'AdaptiveMaxPool3D', slice_idx=1)
    plt.show()


def numpy_save_test(filename, data):
    # Remember to rename filename each time saving
    np.save(filename, data)
    np.savez_compressed(filename, array1=data)
    uncomp_size = os.path.getsize(filename + '.npy') / 1024**3
    comp_size = os.path.getsize(filename + '.npz') / 1024**3
    print('Uncompressed .npy size: {:.6f} GB'.format(uncomp_size))
    print('Compressed .npz size: {:.6f} GB'.format(comp_size))
    print('Compression rate: {:.3f}'.format(uncomp_size / comp_size))


def numpy_load_test(file_1, file_2):
    # If .npy file, single array is returned
    # If .npz file, dictionary-like object is returned containing {filename: array} pair for each file in archive
    # Need allow_pickle=True to load NumPy object arrays
    X_npy = np.load(file_1, allow_pickle=True)
    X_npz = np.load(file_2, allow_pickle=True)
    print('X_npy type: {}    X_npy shape: {}'.format(type(X_npy), X_npy.shape))
    print('X_npz type: {}    X_npz files: {}'.format(type(X_npz), X_npz.files))
    print('X_npz[''array1''] shape: {}'.fomat(X_npz['array1'].shape))


def padding_grid_test(X):
    # Evaluate increase in size of padding all elements in X to largest, all-encompassing grid shape
    print('Size of X_train before padding: {} bytes'.format(sys.getsizeof(X)))
    padded_X, orig_shapes = pad_grids(X)
    print('Size of X_train after padding: {} bytes'.format(sys.getsizeof(padded_X)))
    print('Increase in size: {:.0%}\n'.format(
          (100 * ((sys.getsizeof(padded_X) - sys.getsizeof(X)) / sys.getsizeof(X)))),
    )


def main():
    GON_0 = np.load('/path/to/GON-0.npy')
    contourfshow(GON_0, 'GON-0')  # NEED TO UPDATE
    plt.show()


if __name__ == "__main__":
    main()

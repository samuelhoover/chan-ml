import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from utils.vis_utils import contourfshow, volshow
from utils.data import ZeoStructDataset, PeriodicPadding, ToTensor


# These testing scripts below were used in a deprecated version of my ZeoStructDataset implementation.
# I am leaving these here for archival purposes and in hopes they might be useful.


# Verifying that ZeoStructDataset had the capability to loaded variably sized inputs.
#       Result -- ZeoStructDataset can handle variably sized inputs
def dataset_test(dataset):
    """
    Check to see if ZeoStructDataset is loading energy grid files and targets correctly.

    Args:
        dataset: a preloaded ZeoStructDataset instance
    """

    for i, sample in enumerate(dataset):
        sample = dataset[i]
        zeolite_name = 'I used to pass the zeolite structure name in `sample` but no longer do.'
        print(i, zeolite_name, sample[0].shape, sample[1])

        plt.subplot(2, 2, i + 1)
        contourfshow(sample[0], '{} [kH = {:.3f} mol/kg/MPa]'.format(zeolite_name, sample[1]))
        volshow(sample[0], '{} [kH = {:.3f} mol/kg/MPa]'.format(zeolite_name, sample[1]))

        if i == 3:
            plt.show()
            break


# Verifying that PeriodicPadding could handle variably sized inputs and pad correctly.
#       Results -- PeriodicPadding was able to do both.
def transform_test(dataset):
    """
    Check to see if PeriodicPadding is padding energy grid files correctly.

    Args:
        dataset: a preloaded ZeoStructDataset instance
    """

    idx = np.random.randint(len(dataset))
    sample = dataset[idx]
    pbc = PeriodicPadding()
    
    # Apply periodic boundary conditions on a sample.
    tsfrm_sample = pbc(sample)

    zeolite_name = 'I used to pass the zeolite structure name in `sample` but no longer do.'
    energy_grid, kH = sample[0], sample[1]
    pbc_grid = tsfrm_sample[0]

    # Padded grids should be 2x greater in each direction
    print('Grid size before PBC: {}\nGride size after PBC: {}\n'.format(energy_grid.shape[:3], pbc_grid.shape))
    
    plt.suptitle('{} [kH = {:.3f} mol/kg/MPa]'.format(zeolite_name, kH))
    plt.subplot(1, 2, 1)
    contourfshow(energy_grid, 'Original', slice_idx=1)
    plt.subplot(1, 2, 2)
    contourfshow(pbc_grid, 'PBC', slice_idx=1)
    plt.tight_layout()
    plt.show()


# Verifying if DataLoader could handle mini-batch loading (batch size > 1 < size of full batch) 
# of variably sized inputs via ZeoStructDataset.
#       Result -- DataLoader could not handle variably sized inputs in mini-batches (at least with the collate
#       function I used).
def dataloader_test(dataset):
    """
    Test ZeoStructDataset batch size viability with DataLoader.

    Args:
        dataset: a preloaded ZeoStructDataset instance
    """

    # This collate function did not work in my testing for batch_size > 1.
    def my_collate(batch):
        grids = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return grids, targets


    print('batch_size > 1')
    try:
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=my_collate)
        for idx, sample in enumerate(dataloader):
            zeolite_names = 'I used to pass the zeolite structure name in `sample` but no longer do.'
            for i in range(4):
                print('Zeolite: {}    Grid size: {}    kH: {} mol/kg/MPa'.format(
                    zeolite_names[i], list(sample[0][i].size()), float(sample[1][i])),
                )
                      
            if idx == 0:
                break
        
    except RuntimeError as err:
        print('RuntimeError: ', err)
        print("Can't use DataLoader with non-uniform sized inputs with batch_size > 1.")
    
    else:
        print("DataLoader can be used with non-uniform sized inputs with batch_size > 1.\n")

    print('batch_size = 1')
    try:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=my_collate)
        for idx, sample in enumerate(dataloader):
            zeolite_names = 'I used to pass the zeolite structure name in `sample` but no longer do.'
            for i in range(1):
                print('Zeolite: {}    Grid size: {}    kH: {} mol/kg/MPa'.format(
                    zeolite_names[i], list(sample[1][i].size()), float(sample[2][i])),
                )
                
            if idx == 0:
                break

    except RuntimeError as err:
        print('RuntimeError: ', err)
        print("Can't use DataLoader with non-uniform sized inputs with batch_size = 1.")

    else:
        print("DataLoader can be used with non-uniform sized inputs iff batch_size = 1.\n")


def main():
    return


if __name__ == "__main__":
    main()

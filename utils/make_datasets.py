import csv
import os
import shutil

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import minmax_scale
from tqdm import tqdm


def make_invnorm_dataset(src, dst, IZA=True, PCOD=True, u_lim=8e4):
    """
    Make inverse-normalized energy grid data set from HDF5 files. 
    
    Inverse-normalization == 1 - (x - min)/(max - min) such that
    [lower limit, upper limit] --> [1, 0]

    Args:
        src: path to directory with HDF5 energy grid files
        dst: path to directory to save inverse-normalized energy grid files
        IZA (bool, optional): option to perform on IZA files [default: True]
        PCOD (bool, optional): option to perform on PCOD files [default: True]
        u_lim (optional): upper energy limit [defaults: 8e4 (found from MATLAB testing)]
    """
    
    if IZA & PCOD:
        set = ['IZA', 'PCOD']

    if IZA + PCOD == 1:
        if IZA:
            set = ['IZA']
        
        if PCOD:
            set = ['PCOD']

    for i in set:
        print('Processing... {}'.format(i))
        for file in tqdm(os.listdir(os.path.join(src, i))):
            if file.endswith('.h5'):
                f = h5.File(os.path.join(src, i, file), mode='r')
                grid = np.asarray(f['CH4'])
                grid[grid >= u_lim] = u_lim
                grid = 1 - ((grid - grid.min()) / (grid.max() - grid.min()))
                np.save(os.path.join(dst, file.split('.')[0]), grid)


def make_boltz_weight_dataset(src, dst, IZA=True, PCOD=True, T=1e5):
    """
    Make Boltzmann weighted energy grid data set from HDF5 files.

    Args:
        src: path to source directory with HDF5 energy grid files
        dst: path to save directory
        IZA (bool, optional): option to perform on IZA files [default: True]
        PCOD (bool, optional): option to perform on PCOD files [default: True]
        T (optional): temperature [defaults: 1e5 K]
    """

    if IZA & PCOD:
        set = ['IZA', 'PCOD']

    if IZA + PCOD == 1:
        if IZA:
            set = ['IZA']
        
        if PCOD:
            set = ['PCOD']
    
    print('Making Boltzmann weighted data set at T = {} K\n'.format(T))
    for i in set:
        print('Processing... {}'.format(i))
        for file in tqdm(os.listdir(os.path.join(src, i))):
            if file.endswith('.h5'):
                f = h5.File(os.path.join(src, i, file), mode='r')
                grid = np.asarray(f['CH4'])
                grid = np.exp(-grid / T)
                np.save(os.path.join(dst, file.split('.')[0]), grid)


def split_energy_grid_dataset(src, dst, csv_file, val_split, test_split):
    """
    Split energy grid files and targets into training, validation, and test sets and copy into 
    individual directory for each split.

    Args:
        src: path to source directory for energy grid files
        dst: path to save directory for energy grid files
        csv_file: path to file containing zeolite structure names and targets
        val_split: proportion of data set to be included in validation set
        test_split: proportion of data set to be included in test set
    """

    df = pd.read_csv(csv_file)  # load CSV file containing zeolite structure names and targets into a DataFrame

    # Split DataFrame into training, validation, and test sets
    train, val, test = np.split(
        df.sample(frac=1), 
        [int((1 - val_split - test_split) * len(df)), int((1 - test_split) * len(df))],
    )

    # Copy targets from source file into individual CSV file for each split
    train.to_csv(os.path.join(dst, 'train/train_kH.csv'), index=False)
    val.to_csv(os.path.join(dst, 'val/val_kH.csv'), index=False)
    test.to_csv(os.path.join(dst, 'test/test_kH.csv'), index=False)

    # Copy energy grid files from source directory into individual directories for each split
    for zeo in train['Zeolite']:
        shutil.copy2(
            os.path.join(src, zeo + '.npy'), 
            os.path.join(dst, 'train', zeo + '.npy'),
        )

    for zeo in val['Zeolite']:
        shutil.copy2(
            os.path.join(src, zeo + '.npy'), 
            os.path.join(dst, 'val', zeo + '.npy'),
        )

    for zeo in test['Zeolite']:
        shutil.copy2(
            os.path.join(src, zeo + '.npy'), 
            os.path.join(dst, 'test', zeo + '.npy'),
        )


def extract_channel_info(src, dst, csv_mode='w', print_header=True):
    """
    Extract zeolite channel geometry info from raw text files.

    *** Sample of raw text file ***
    ZEO-0.chan   2 channels identified of dimensionality 2 3 
    Channel  0  12.5368  6.25191  12.5368
    Channel  1  9.4434  4.2921  9.4434
    *** End of raw text file ***

    Args:
        src: path to source directory
        dst: path to destination file
        csv_mode (optional): how to handle CSV file ('w', write; 'r', read; 'a', append) [default: w]
        print_header (bool, optional): option to print header for CSV file [default: True]
    """

    with open(dst, mode=csv_mode) as csv_file:
        writer = csv.writer(csv_file)

        src_files = os.listdir(src)
        src_files.sort(key=str.lower)  # sort directory alphabetically
        for file in tqdm(src_files):

            # Max number of channels is 16, hence vectors length = 16 for succeeding 4 descriptors
            dims = np.zeros(16).astype(np.int)  # channel dimensionality
            l_i_sph = np.zeros(16).astype(np.float)  # largest included sphere
            l_f_sph = np.zeros(16).astype(np.float)  # largest free sphere
            l_f_sph_path = np.zeros(16).astype(np.float)  # largest included sphere along free sphere path

            chan_vec = np.zeros(66, dtype=object)  # vector containing all zeolite channel information

            f = open(os.path.join(src, file), 'r')
            for i, line in enumerate(f):
                if i == 0:  # get info from first line
                    zeo = file.split('.')[0]  # zeolite structure name
                    num_channels = int(line.split(' ')[3])  # number of channels

                    _dims = line.rstrip('\n').split(' ')[-num_channels-1:-1]  # get block of text for dimensionality
                    for idx_dim, _dim in enumerate(_dims):
                        dims[idx_dim] = _dim

                    chan_vec[1] = num_channels
                    chan_vec[0] = zeo
                    chan_vec[-64:-48] = dims
                    
                if num_channels == 0:  # if no channels exist, exit file analysis loop
                    break
                
                if i != 0:  # get info from remaining lines, if applicable
                    geo_line = line.replace('\n', '').split('  ')
                    l_i_sph[i-1] = geo_line[-3]
                    l_f_sph[i-1] = geo_line[-2]
                    l_f_sph_path[i-1] = geo_line[-1]
                
            chan_vec[-48:-32] = l_i_sph
            chan_vec[-32:-16] = l_f_sph
            chan_vec[-16:] = l_f_sph_path
            writer.writerow(chan_vec)

            f.close()

    csv_file.close()


def extract_psd_info(src, dst, csv_mode='w', print_header=True):
    """
    Extract zeolite pore size distribution info from raw text files.

    *** Sample of raw text file ***
    Pore size distribution histogram
    Bin size (A): 0.1
    Number of bins: 1000
    From: 0
    To: 100
    Total samples: 500000
    Accessible samples: 19046
    Fraction of sample points in node spheres: 0.038092
    Fraction of sample points outside node spheres: 0

    Bin Count Cumulative_dist Derivative_dist
    0.0 0 1 0
    0.1 0 1 0
    ...
    ...
    ...
    99.9 0 0 0
    *** End of raw text file ***

    Args:
        src: path to source directory
        dst: path to destination file
        csv_mode (optional): how to handle CSV file ('w', write; 'r', read; 'a', append [default: w])
        print_header (bool, optional): option to print header for CSV file [default: True]
    """

    with open(dst, mode=csv_mode) as csv_file:
        writer = csv.writer(csv_file)
        if print_header:
            writer.writerow(['zeolite', 'mean', '1st_quartile', 'median', '3rd_quartile'])
        
        src_files = os.listdir(src)
        src_files.sort(key=str.lower)  # sort directory alphabetically
        for file in tqdm(src_files):
            _zeo = file.split('/')[-1]
            zeo = _zeo.split('.')[0]

            f = open(os.path.join(src, file), 'r')  # read in text file to obtain number of accessible samples
            for i, line in enumerate(f):
                if i == 6:
                    num_acc = int(line.split(' ')[-1])
                    break  # stop reading after the 7th line (line indexing starts at 0, hence i=6 == line # 7)

            if num_acc == 0:
                writer.writerow([zeo, 0, 0, 0, 0])

            else:
                data = np.loadtxt(os.path.join(src, file), skiprows=11)  # read in *only* binned data
                mean = np.dot(data[:, 0].T, data[:, 1]) / num_acc
                cumsum_count = np.cumsum(data[:, 1])
                quarts = np.zeros(3)
                for i, q in enumerate([1, 2, 3]):  # quartiles
                    q_idx = (num_acc + 1) * q / 4
                    if not q_idx.is_integer():  # if quartile index is not a whole number
                        upper_idx = np.where(cumsum_count > np.ceil(q_idx))[0][0]
                        lower_idx = np.where(cumsum_count > np.floor(q_idx))[0][0]
                        if data[upper_idx, 0] != data[lower_idx, 0]:  # if quartile value falls between two unique numbers
                            quarts[i] = (data[upper_idx, 0] + data[lower_idx, 0]) / 2

                        else:
                            quarts[i] = data[np.where(cumsum_count > q_idx)[0][0], 0]

                    else:
                        quarts[i] = data[np.where(cumsum_count > q_idx)[0][0], 0]

                writer.writerow([zeo, mean, quarts[0], quarts[1], quarts[2]])
            
            f.close()
    
    csv_file.close()


def make_channel_dataset():
    """Extract IZA and PCOD channel geometry information and combine into CSV file."""

    print('Extracting IZA channel info ...')
    extract_channel_info(
        '/path/to/IZA/channel-info/',
        '/path/to/destination/dataset.csv',
    )
    print('IZA extraction complete.')

    print('Extracting PCOD channel info ...')
    extract_channel_info(
        '/path/to/PCOD/channel-info/',
        '/path/to/destination/dataset.csv',
        csv_mode='a',
        print_header=False,
    )
    print('PCOD extraction complete.')


def remove_last_column(src, dst):
    """
    I don't remember what this was for but I had to remove the last column (i.e. last few characters from each line;
    I think it was whitespace ?) of something.

    Args:
        src: path to source file
        dst: path to save file
    """

    with open(dst, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        f = open(src, mode='r')
        for i, line in enumerate(f):
            if i == 0:
                writer.writerow(line.split(' '))
            
            if i > 0:
                writer.writerow(line.split(' ')[:-1])
            
    csv_file.close()


def make_full_chemistry_channel_dataset():
    """Load and merge chemistry and channel info into single data set."""

    # Load IZA and PCOD CSV files, rename first `IZA`/`PCOD` columns to 'zeolite', add `set` column for which
    # dataset the zeolites belong to
    df_IZA = pd.read_csv('/path/to/IZA/dataset.csv', dtype={'IZA': str})
    df_PCOD = pd.read_csv('/path/to/PCOD/dataset.csv', dtype={'PCOD': str})
    df_IZA = df_IZA.rename(columns={'IZA': 'zeolite'})
    df_PCOD = df_PCOD.rename(columns={'PCOD': 'zeolite'})
    df_IZA.insert(1, 'set', 'IZA')
    df_PCOD.insert(1, 'set', 'PCOD')

    # Stack both DataFrames vertically
    df_chem = pd.concat([df_IZA, df_PCOD], axis='index').reset_index(drop=True)

    # Load zeolite channel info, add `set` column for which dataset the zeolites belong to
    df_chan = pd.read_csv('/path/to/PB/screening/dataset.csv', dtype={'zeolite': str})
    df_chan.insert(1, 'set', 'PCOD')
    df_chan.loc[:401, 'set'] = 'IZA'

    # Merge PB screening data (`df`) with channel info (`each_zeolite`)
    df = df_chem.merge(df_chan.iloc[:, :-1], left_on=['zeolite', 'set'], right_on=['zeolite', 'set'])

    # Swap columns to keep (min, max) convention
    cols = list(df)
    cols[-6], cols[-7] = cols[-7], cols[-6]
    df.columns = cols

    # Rename columns for brevity
    df = df.rename(
        columns={
            'Square_end2end_distance_C18': 'SETE_C18',
            'Largest_Cavity_Diameter_min': 'LCD_min',
            'Pore_Limiting_Diameter_min': 'PLD_min',
            'LCD_along_free_sphere_path_max': 'LCD_free_max',
        },
    )

    df.to_csv('/path/to/destination/dataset.csv', index=False)  # save to CSV file


def make_preprocessed_minmaxnorm():
    """Load chemistry-channel info and perform min-max normalization over features."""

    df = pd.read_csv('/path/to/full/dataset.csv', low_memory=False)
    df = df[df.num_channels != 0]  # remove zeolites with no channels

    # Min-max normalization
    feature_cols = ['dim_C18', 'geometrical_dimension', 'LCD_min', 'LCD_max', 'PLD_min',
                    'PLD_max', 'LCD_free_min', 'LCD_free_max', 'num_channels']
    df_features = minmax_scale(df[feature_cols], feature_range=(0, 1), axis=0)  # replaced with sklearn's min-max normalization

    df_to_keep = pd.concat([df_features, df['kH_C18']], axis='columns')  # add targets
    df_to_keep.to_csv('/path/to/destination/dataset.csv', index=False)  # save to CSV file


def main():
    make_boltz_weight_dataset(
        src='/path/to/src',
        dst='/path/to/dst',
        IZA=True,
        PCOD=True,
    )


if __name__ == "__main__":
    main()

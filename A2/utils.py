import os

import numpy as np
import pandas as pd
from tqdm import tqdm

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def read_in_all_csv_in_dir(dir_path: str):
    """Reads in all .csv files in a directory and returns a list of pandas DataFrames."""
    all_data = []
    for filename in tqdm(os.listdir(dir_path)):
        if filename.endswith('.csv') and not os.path.exists(filename.replace('.csv', '.npy')):
            file_path = os.path.join(dir_path, filename)
            array = pd.read_csv(file_path).to_numpy()
            array.tofile(file_path.replace('.csv', '.npy'))
            all_data.append(array)
    return all_data

def read_in_all_npy_in_dir(dir_path: str):
    """Reads in all .npy files in a directory and returns a list of numpy arrays."""
    all_data = []
    for filename in tqdm(os.listdir(dir_path)):
        if filename.endswith('.npy'):
            file_path = os.path.join(dir_path, filename)
            array = np.load(file_path, allow_pickle=True)
            all_data.append(array)
    return all_data

# Logger
import logger

_logger = logger.get_logger('datasets.py', logger.INFO)

# Imports
import torch
import torch.nn as nn
import torch.utils.data as td

import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt



# Custom data loader
class PeanoDataset(td.Dataset):
    def __init__(self, size, overlap):
        df = pd.read_csv('datasets/co2_peano_no_weekend.csv')
        db = df[['_value']].values

        self.data = create_windows(db, size, 1, overlap)
        _logger.info(f'Loaded dataset: ({self.data.shape})')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        x_raw = self.data[idx]

        x = torch.tensor(x_raw).T.float()
        y = x.clone()

        return x, y

class UciDataset(td.Dataset):
    def __init__(self, size, overlap):
        db = np.load('datasets/UCI-HAR.npy')
        
        self.data = create_windows(db, size, 6, overlap)
        _logger.info(f'Loaded dataset: ({self.data.shape})\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        x_raw = self.data[idx]

        x = torch.tensor(x_raw).T.float()
        y = x.clone()

        return x, y

class MotionSenseDataset(td.Dataset):
    def __init__(self, size, overlap):
        db = np.load('datasets/MotionSense.npy')
        
        self.data = create_windows(db, size, 6, overlap)
        _logger.info(f'Loaded dataset: ({self.data.shape})\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        x_raw = self.data[idx]

        x = torch.tensor(x_raw).T.float()
        y = x.clone()

        return x, y

class WISDMDataset(td.Dataset):
    def __init__(self, size, overlap):
        db = np.load('datasets/WISDM-General.npy')
        
        self.data = create_windows(db, size, 6, overlap)
        _logger.info(f'Loaded dataset: ({self.data.shape})\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        x_raw = self.data[idx]

        x = torch.tensor(x_raw).T.float()
        y = x.clone()

        return x, y



# Utility functions
## Windows
def create_windows(db, size, delimiter, overlap = 0.0):
    _logger.info(f'Creating windows - Started')

    windows = []

    # Iteration step 
    step = size - int(size * overlap)
    i = 0
    total_db_len = len(db)

    # Loop data
    while (i + size) < total_db_len:
        # Get data
        curr = np.array(
            db[i:(i + size), :delimiter]
        )
        
        # Push to output
        windows.append(curr)

        # Step to next window
        i += step

    _logger.info(f'Creating windows - Terminated\n')
    
    return np.array(windows)
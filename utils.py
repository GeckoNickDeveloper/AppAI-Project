# Logger
import logger

_logger = logger.get_logger('UTILS', logger.INFO)

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

## Progress Bar
def progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 80, fill = '▬', to_fill = ' ', end = '\r'):
    # Defualt prefix
    if prefix == '':
        prefix = f'{iteration}/{total}'
    
    # Compute percentage string
    percentage = 100.0 * (iteration / float(total))
    percent = ('{0:.' + str(decimals) + 'f}').format(percentage)

    # Compute progress bar string
    filled_len = int(length * iteration // total)
    bar = '\x1B[38;2;255;0;255m' + fill * filled_len + '\033[0m' + to_fill * (length - filled_len)

    # Print the progress bar
    print(f'\r{prefix} [{bar}] {percent}% {suffix}', end = end)
    
    # New Line on complete
    if iteration == total:
        print()

## Train function
def train(model, dataloader, criterion, optimizer, device, show_progress = True):
    model.train()
    run_loss = 0.0

    # Counters (progressbar)
    current_batch = 1
    batches = len(dataloader)

    # Actual loop
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # print(inputs[0])
        # print(labels[0])
        # 1/0

        optimizer.zero_grad()

        outputs = model(inputs)
        # print(outputs.size())
        # print(outputs - inputs)

        loss = criterion(outputs, labels)

        # with torch.autograd.set_detect_anomaly(True):
            # loss.backward()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

        optimizer.step()

        run_loss += loss.item() * inputs.size(0)

        # Progress Bar
        if show_progress:
            progress_bar(current_batch, batches, )
        current_batch += 1

    return run_loss / len(dataloader.dataset)

## Evaluation function
def evaluate(model, dataloader, criterion, device, show_progress = True):
    model.eval()
    run_loss = 0.0

    with torch.no_grad():
        # Counters (progressbar)
        current_batch = 1
        batches = len(dataloader)
    
        # Actual loop
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            run_loss += loss.item() * inputs.size(0)

            # Progress Bar
            if show_progress:
                progress_bar(current_batch, batches, )
            current_batch += 1


    return run_loss / len(dataloader.dataset)

## Determinism
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

## Plotting
def plot_har(x_true: np.ndarray, x_pred: np.ndarray, filename: str):
    # Plot colors
    colors = [
        '#118ab2',
        '#ffd166',
        '#ef476f',
        #'#acacac',
        #'#06d6a0',
    ]
    
    
    
    # True Acc
    fig, ax = plt.subplots(1, figsize=(10, 8))
    
    ax.plot(x_true[:,0], color = colors[0], label = 'X')
    ax.plot(x_true[:,1], color = colors[1], label = 'Y')
    ax.plot(x_true[:,2], color = colors[2], label = 'Z')
    
    ax.set_title('True Acc')
    ax.grid()
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(f'plots/{filename}-true-acc.png', format='png')
    plt.savefig(f'plots/{filename}-true-acc.eps', format='eps')
    
    
    
    # True Gyro
    fig, ax = plt.subplots(1, figsize=(10, 8))
    
    ax.plot(x_true[:,3], color = colors[0], label = 'X')
    ax.plot(x_true[:,4], color = colors[1], label = 'Y')
    ax.plot(x_true[:,5], color = colors[2], label = 'Z')
    
    ax.set_title('True Gyro')
    ax.grid()
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(f'plots/{filename}-true-gyro.png', format='png')
    plt.savefig(f'plots/{filename}-true-gyro.eps', format='eps')
    
    
    
    # Pred Acc
    fig, ax = plt.subplots(1, figsize=(10, 8))
    
    ax.plot(x_pred[:,0], color = colors[0], label = 'X')
    ax.plot(x_pred[:,1], color = colors[1], label = 'Y')
    ax.plot(x_pred[:,2], color = colors[2], label = 'Z')
    
    ax.set_title('Pred Acc')
    ax.grid()
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(f'plots/{filename}-pred-acc.png', format='png')
    plt.savefig(f'plots/{filename}-pred-acc.eps', format='eps')
    
    
    
    # Pred Gyro
    fig, ax = plt.subplots(1, figsize=(10, 8))
    
    ax.plot(x_pred[:,3], color = colors[0], label = 'X')
    ax.plot(x_pred[:,4], color = colors[1], label = 'Y')
    ax.plot(x_pred[:,5], color = colors[2], label = 'Z')
    
    ax.set_title('Pred Gyro')
    ax.grid()
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(f'plots/{filename}-pred-gyro.png', format='png')
    plt.savefig(f'plots/{filename}-pred-gyro.eps', format='eps')
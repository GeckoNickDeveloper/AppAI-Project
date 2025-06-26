# Imports
import torch
import torch.nn as nn
import torch.utils.data as td

import numpy as np
import pandas as pd

# Custom data loader
class PeanoDataset(td.Dataset):
    def __init__(self, size, overlap):
        df = pd.read_csv('datasets/co2_peano_no_weekend.csv')
        db = df[['_value']].values

        self.data = create_windows(db, size, 1, overlap)
        print(self.data.shape)
    
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
        
        self.data = create_windows(db, size, 3, overlap)
        print(self.data.shape)

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
    windows = []

    # Iteration step 
    step = size - int(size * overlap)
    i = 0
    total_db_len = len(db)
    
    print('[create_windows] Started')

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

    print('[create_windows] Terminated successfully')
    
    return np.array(windows)

## Progress Bar
def progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 80, fill = '=', to_fill = ' ', end = '\r'):
    # Defualt prefix
    if prefix == '':
        prefix = f'{iteration}/{total}'
    
    # Compute percentage string
    percentage = 100.0 * (iteration / float(total))
    percent = ('{0:.' + str(decimals) + 'f}').format(percentage)

    # Compute progress bar string
    filled_len = int(length * iteration // total)
    bar = fill * filled_len + to_fill * (length - filled_len)

    # Print the progress bar
    print(f'\r{prefix} [{bar}] {percent}% {suffix}', end = end)
    
    # New Line on complete
    if iteration == total:
        print()

## Train function
def train(model, dataloader, criterion, optimizer, device):
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
        progress_bar(current_batch, batches, )
        current_batch += 1

    return run_loss / len(dataloader.dataset)

## Evaluation function
def evaluate(model, dataloader, criterion, device):
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
            progress_bar(current_batch, batches, )
            current_batch += 1


    return run_loss / len(dataloader.dataset)



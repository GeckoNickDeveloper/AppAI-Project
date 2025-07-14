# Logger
import logger

_logger = logger.get_logger('UTILS', logger.INFO)

# Imports
import torch
import torch.nn as nn

import numpy as np
import random
import os

import matplotlib.pyplot as plt

# Build directory by given path
def build_directories(path, verbose: bool = True):
    directories = path.split('/')
    building = ''

    for directory in directories:
        building += directory + '/'

        if verbose:
            _logger.info(f'Building {building}')

        if not os.path.exists(building):
            os.mkdir(building)

# Progress Bar
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

# Train function
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

# Evaluation function
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
                progress_bar(current_batch, batches)
            current_batch += 1


    return run_loss / len(dataloader.dataset)

# Predict
def predict(model, dataloader, device, show_progress = True):
    # Set to eval mode
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        # Counters (progressbar)
        current_batch = 1
        batches = len(dataloader)
    
        # Actual loop
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            output = model(inputs)
            predictions.append(output.cpu().numpy().transpose(0,2,1))
            targets.append(labels.numpy().transpose(0,2,1))

            # Progress Bar
            if show_progress:
                progress_bar(current_batch, batches)
            current_batch += 1

    # Stack all batches
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Return (y_pred, y_true)
    return (
        predictions.reshape((predictions.shape[0], -1)),
        targets.reshape((targets.shape[0], -1))
    )

# Determinism
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

# Plotting
def plot_acc_gyro(x_true: np.ndarray, x_pred: np.ndarray, filename: str):
    # Plot colors
    colors = [
        '#118ab2',
        '#ffd166',
        '#ef476f',
        
        '#2ba811',
        '#ff7438',
        '#661dff',
    ]
    
    
    
    # Acc
    fig, ax = plt.subplots(1, figsize=(15, 10))
    
    ax.plot(x_true[:,0], linestyle = '--', color = colors[0], label = 'Original X')
    ax.plot(x_pred[:,0], linestyle = '-', color = colors[3], label = 'Reconstruced X')
    
    ax.plot(x_true[:,1], linestyle = '--', color = colors[1], label = 'Original Y')
    ax.plot(x_pred[:,1], linestyle = '-', color = colors[4], label = 'Reconstruced Y')
    
    ax.plot(x_true[:,2], linestyle = '--', color = colors[2], label = 'Original Z')
    ax.plot(x_pred[:,2], linestyle = '-', color = colors[5], label = 'Reconstruced Z')
    
    
    # ax.set_title('Accelerometer')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.grid()
    ax.legend(fontsize = 18, loc = 'lower center', ncol = 3)

    fig.tight_layout()
    plt.savefig(f'{filename}-acc.png', format='png')
    plt.savefig(f'{filename}-acc.eps', format='eps')



    # Gyro
    fig, ax = plt.subplots(1, figsize=(15, 10))

    ax.plot(x_true[:,3], linestyle = '--', color = colors[0], label = 'Original X')
    ax.plot(x_pred[:,3], linestyle = '-', color = colors[3], label = 'Reconstructed X')
    
    ax.plot(x_true[:,4], linestyle = '--', color = colors[1], label = 'Original Y')
    ax.plot(x_pred[:,4], linestyle = '-', color = colors[4], label = 'Reconstructed Y')
    
    ax.plot(x_true[:,5], linestyle = '--', color = colors[2], label = 'Original Z')
    ax.plot(x_pred[:,5], linestyle = '-', color = colors[5], label = 'Reconstructed Z')
    
    # ax.set_title('Gyroscope')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.grid()
    ax.legend(fontsize = 18, loc = 'lower center', ncol = 3)
    
    # Plots
    fig.tight_layout()
    plt.savefig(f'{filename}-gyro.png', format='png')
    plt.savefig(f'{filename}-gyro.eps', format='eps')
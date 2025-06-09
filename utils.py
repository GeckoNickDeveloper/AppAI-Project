# Imports
import torch
import torch.nn as nn
import torch.utils.data as td

# Custom data loader
class HARDataset(td.Dataset):
    """
        TODO: doc
    """
    def __init__(self, path):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        return x, y


# Utility functions
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
        
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        run_loss += loss.item() * inputs.size(0)

        # Progress Bar
        utils.progress_bar(current_batch, batches, )
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
            utils.progress_bar(current_batch, batches, )
            current_batch += 1


    return run_loss / len(dataloader.dataset)



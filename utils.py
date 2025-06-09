# Imports
import torch
import torch.nn as nn
import torch.utils.data as td



# Custom data loader










# Utility functions
## Train function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    run_loss = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        run_loss += loss.item() * inputs.size(0)

    return run_loss / len(dataloader.dataset)

## Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    run_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            run_loss += loss.item() * inputs.size(0)

    return run_loss / len(dataloader.dataset)





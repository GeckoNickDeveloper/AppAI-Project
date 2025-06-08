# Imports
import torch
import torch.nn as nn
import numpy as np

import ae



# Settings
## Globals
SEED = 42069

## Model
INPUTS_SHAPE = (6, 3000) # 1m at 50Hz (6 channels data)

## Training
EPOCHS = 100
BATCH_SIZE = 64

## Paths
MODEL_FILENAME = f'models/AutoEncoder_e{EPOCHS}_bs{BATCH_SIZE}_seed{SEED}.pt'



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



# Load dataset
# TODO: Create dataloaders



# Setup model, criterion and optimizer
## Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Model
model = ae.AutoEncoder().to(device)

## Criterion
# criterion = nn.L1Loss() # MAE
criterion = nn.MSELoss() # MSE

## Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)



# Train
for i in range(EPOCHS):
    # TODO: Create dataloaders
    train_loss = train(model, train_loader, criterion, optimizer, device)
    eval_loss = evaluate(model, test_loader, criterion, device)
    print(f'Epoch [{i+1}/{EPOCHS}] - Train: {train_loss} - Eval: {eval_loss}')



## Save model
torch.save(model.state_dict(), MODEL_FILENAME)

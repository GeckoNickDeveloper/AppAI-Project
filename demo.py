# Imports
import torch
import torch.nn as nn
import numpy as np

import ae
import utils


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

# Load dataset
# TODO: Import data


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
    train_loss = utils.train(model, train_loader, criterion, optimizer, device)
    eval_loss = utils.evaluate(model, test_loader, criterion, device)
    print(f'Epoch [{i+1}/{EPOCHS}] - Train: {train_loss} - Eval: {eval_loss}')


# Model files
## Save model
torch.save(model, MODEL_FILENAME)

## Load model
model = torch.load(MODEL_FILENAME)

# Evaluate model
eval_loss = utils.evaluate(model, test_loader, criterion, device)
print(f'Eval: {eval_loss}')


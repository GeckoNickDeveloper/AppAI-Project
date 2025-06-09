# Imports
import torch
import torch.nn as nn
import torch.utils.data as td

import numpy as np

import ae
import utils



# Settings
## Globals
SEED = 42069
TRAIN_SIZE = 0.8

## Model
INPUTS_SHAPE = (6, 3000) # 1m at 50Hz (6 channels data)

## Training
EPOCHS = 100
BATCH_SIZE = 64

## Paths
DATASET_PATH = f'datasets/UCI-HAR.npy'
MODEL_PATH = f'models/AutoEncoder_e{EPOCHS}_bs{BATCH_SIZE}_seed{SEED}.pt'



# Dataset
dataset = utils.HARDataset(DATASET_PATH)

## Train/Test Split
train_size = int(TRAIN_SIZE * len(dataset))
test_size = len(dataset) - train_size

train_set, test_set = td.random_split(dataset, [train_size, test_size])

## Loaders
train_loader = td.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
test_loader = td.DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False)



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
    train_loss = utils.train(model, train_loader, criterion, optimizer, device)
    eval_loss = utils.evaluate(model, test_loader, criterion, device)
    print(f'Epoch [{i+1}/{EPOCHS}] - Train: {train_loss} - Eval: {eval_loss}')



# Model files
## Save model
torch.save(model, MODEL_PATH)

## Load model
model = torch.load(MODEL_PATH)

# Evaluate model
eval_loss = utils.evaluate(model, test_loader, criterion, device)
print(f'Eval: {eval_loss}')


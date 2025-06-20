# Imports
import torch
import torch.nn as nn
import torch.utils.data as td

import numpy as np

import matplotlib.pyplot as plt

import ae
import utils



# Settings
## Globals
SEED = 42069
TRAIN_SIZE = 0.8

## Model
INPUTS_SHAPE = (6, 100) # 2s at 50Hz (6 channels data)

## Dataset
OVERLAP = 0.0

## Training
EPOCHS = 50
BATCH_SIZE = 64

## Paths
MODEL_PATH = f'models/AutoEncoder_e{EPOCHS}_bs{BATCH_SIZE}_seed{SEED}.pt'



# Dataset
dataset = utils.UciDataset(INPUTS_SHAPE[1], OVERLAP)

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
model = ae.AutoEncoder(INPUTS_SHAPE[0]).to(device)

## Criterion
criterion = nn.L1Loss() # MAE
# criterion = nn.MSELoss() # MSE

## Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)



# Train
for i in range(EPOCHS):
    train_loss = utils.train(model, train_loader, criterion, optimizer, device)
    eval_loss = utils.evaluate(model, test_loader, criterion, device)
    print(f'Epoch [{i+1}/{EPOCHS}] - Train: {train_loss} - Eval: {eval_loss}\n')



# Model files
## Save model
torch.save(model, MODEL_PATH)

## Load model
model = torch.load(MODEL_PATH, weights_only = False)

# Evaluate model
eval_loss = utils.evaluate(model, test_loader, criterion, device)
print(f'Eval: {eval_loss}')



# Plot a random window
batch = next(iter(test_loader))
x, _ = batch
x = x.to(device)
model.eval()
with torch.no_grad():
    xp = model(x)


fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].plot(x[0].T[:,:3], label='True')
ax[0].grid()

ax[1].plot(xp[0].T[:,:3], label='Pred')
ax[1].grid()

fig.legend()
plt.show()










# Logger
import logger

_logger = logger.get_logger('DEMO', logger.INFO)

# Imports
_logger.info('Importing libraries\n')

import torch
import torch.nn as nn
import torch.utils.data as td

import time

import ae
import datasets as ds
import utils



# Settings
## Globals
SEED = 42069
TRAIN_SIZE = 0.8

## Model
# INPUTS_SHAPE = (3, 3000) # 1m @50Hz (3 channels data)
INPUTS_SHAPE = (6, 100) # 2s @50Hz (6 channels data)
# INPUTS_SHAPE = (1, 300) # 25h of data @.0033Hz (1 channel data)

## Dataset
OVERLAP = 0.0

## Training
EPOCHS = 5
BATCH_SIZE = 64

## Paths
MODEL_PATH = f'models/AutoEncoder_e{EPOCHS}_bs{BATCH_SIZE}_seed{SEED}.pt'



# Set Determinism
_logger.info('Setting seed for determinism\n')

generator = torch.Generator()
generator.manual_seed(SEED)

utils.set_seed(SEED)



# Dataset
_logger.info('Loading dataset\n')

dataset = ds.UciDataset(INPUTS_SHAPE[1], OVERLAP)
# dataset = ds.PeanoDataset(INPUTS_SHAPE[1], OVERLAP)

## Train/Test Split
train_size = int(TRAIN_SIZE * len(dataset))
test_size = len(dataset) - train_size

_logger.info('Train-Test Split\n')
train_set, test_set = td.random_split(dataset, [train_size, test_size])

## Loaders
train_loader = td.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 1, worker_init_fn = utils.seed_worker, generator = generator)
test_loader = td.DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 1, worker_init_fn = utils.seed_worker, generator = generator)



# Setup model, criterion and optimizer
## Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Model
_logger.info('Creating model\n')

model = ae.AutoEncoder(INPUTS_SHAPE[0], INPUTS_SHAPE[0]).to(device)

## Criterion
# criterion = nn.L1Loss() # MAE
criterion = nn.MSELoss() # MSE

## Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

## Summary
_logger.info('====== SUMMARY ======')
encoder_params = sum(p.numel() for p in model.encoder.parameters())
_logger.info(f'Encoder parameters: {format(encoder_params, ",")}')
decoder_params = sum(p.numel() for p in model.decoder.parameters())
_logger.info(f'Decoder parameters: {format(decoder_params, ",")}')
total_params = sum(p.numel() for p in model.parameters())
_logger.info(f'Total parameters: {format(total_params, ",")}\n')



# Train
_logger.info('Train started\n')
print('============ Train Started')
for i in range(EPOCHS):
    print(f'Epoch [{i+1}/{EPOCHS}]')
    
    epoch_start_time = int(time.time() * 1000)
    train_loss = utils.train(model, train_loader, criterion, optimizer, device, show_progress = True)
    epoch_end_time = int(time.time() * 1000)
    
    eval_loss = utils.evaluate(model, test_loader, criterion, device, show_progress = False)
    print(f'({epoch_end_time - epoch_start_time} ms) Train: {train_loss} - Eval: {eval_loss}\n')



# Model files
## Save model
torch.save(model, MODEL_PATH)

## Load model
model = torch.load(MODEL_PATH, weights_only = False)

# Evaluate model
eval_loss = utils.evaluate(model, test_loader, criterion, device)
print(f'Eval: {eval_loss}')



# Plot
## Get one batch
batch = next(iter(test_loader))
x, _ = batch

## Move to device
x = x.to(device)

## Perform inference
model.eval()
with torch.no_grad():
    xp = model(x)

## Generate plots
# for i in range(x.size(0)):
#     utils.plot_har(x.cpu()[i].T, xp.cpu()[i].T, f'plot-{i}')

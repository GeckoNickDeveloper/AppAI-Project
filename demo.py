# Logger
import logger

_logger = logger.get_logger('DEMO', logger.DEBUG)

# Imports
_logger.info('Importing libraries\n')

import torch
import torch.nn as nn
import torch.utils.data as td

import time
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

import ae
import datasets as ds
import utils



# Settings
CMD_PARAM = True

## Globals
SEED = 42069
TRAIN_SIZE = 0.8

## Model
# INPUTS_SHAPE = (6, 100) # 2s @50Hz (6 channels data)
# INPUTS_SHAPE = (6, 3000) # 2s @50Hz (6 channels data)
INPUTS_SHAPE = (6, 100) # 2s @50Hz (6 channels data)

EMBEDDING_CHANNELS = INPUTS_SHAPE[0]
FILTERS = 8

## Dataset
DATASET_NAME = 'uci'
OVERLAP = 0.5

## Training
EPOCHS = 100
BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 10

## COMMAND LINE OVERRIDE
if CMD_PARAM:
    # Model
    INPUTS_SHAPE = (6, int(sys.argv[1]))
    EMBEDDING_CHANNELS = int(sys.argv[2])
    FILTERS = int(sys.argv[3])
    
    # Training
    EPOCHS = int(sys.argv[4])
    BATCH_SIZE = int(sys.argv[5])
    
    # General
    SEED = int(sys.argv[6])
    
    # Dataset
    DATASET_NAME = sys.argv[7]

## Dataset override
if DATASET_NAME.lower() not in ['uci', 'motionsense', 'wisdm']:
    DATASET_NAME = 'uci'

## Paths
MODEL_PATH = f'./models/{DATASET_NAME}'
MODEL_NAME = f'AutoEncoder_f{FILTERS}_ws{INPUTS_SHAPE[1]}_ch{INPUTS_SHAPE[0]}_e{EPOCHS}_bs{BATCH_SIZE}_seed{SEED}'
MODEL_FILENAME = f'{MODEL_PATH}/{MODEL_NAME}.pt'

LOG_PATH = f'./log/{DATASET_NAME}'
LOG_NAME = f'sweep_ws{INPUTS_SHAPE[1]}_ch{INPUTS_SHAPE[0]}_o{OVERLAP}'
LOG_FILENAME = f'{LOG_PATH}/{LOG_NAME}.log'

# Build directories
utils.build_directories(MODEL_PATH)
utils.build_directories(LOG_PATH)

# Set Determinism
_logger.info('Setting seed for determinism\n')

generator = torch.Generator()
generator.manual_seed(SEED)

utils.set_seed(SEED)



# Dataset
_logger.info('Loading dataset\n')

if DATASET_NAME == 'uci':
    dataset = ds.UciDataset(INPUTS_SHAPE[1], OVERLAP)
elif DATASET_NAME == 'motionsense':
    dataset = ds.MotionSenseDataset(INPUTS_SHAPE[1], OVERLAP)
elif DATASET_NAME == 'wisdm':
    dataset = ds.WISDMDataset(INPUTS_SHAPE[1], OVERLAP)
else:
    dataset = ds.UciDataset(INPUTS_SHAPE[1], OVERLAP)

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

model = ae.AutoEncoder(INPUTS_SHAPE[0], FILTERS, EMBEDDING_CHANNELS).to(device)

## Best model
best_model = ae.AutoEncoder(INPUTS_SHAPE[0], FILTERS, EMBEDDING_CHANNELS)


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

## Counters
patience_counter = 0
best_loss = float('inf')

## Training loop
for i in range(EPOCHS):
    # Early stopping
    if patience_counter == EARLY_STOP_PATIENCE:
        _logger.warning(f'Early Stopping: training ended in {i} epochs')
        break

    print(f'Epoch [{i+1}/{EPOCHS}]')
    
    epoch_start_time = int(time.time() * 1000)
    train_loss = utils.train(model, train_loader, criterion, optimizer, device, show_progress = True)
    epoch_end_time = int(time.time() * 1000)
    
    eval_loss = utils.evaluate(model, test_loader, criterion, device, show_progress = False)
    print(f'({epoch_end_time - epoch_start_time} ms) Train: {train_loss} - Eval: {eval_loss}\n')

    # Save best model only
    if eval_loss < best_loss:
        # Reset patience counter
        patience_counter = 0
        # Save best
        best_model.load_state_dict(model.state_dict())
    else:
        patience_counter += 1



# Model files
## Save model
torch.save(best_model, MODEL_FILENAME)

## Load model
model = torch.load(MODEL_FILENAME, weights_only = False)

# Evaluate model
eval_loss = utils.evaluate(model, test_loader, criterion, device)
print(f'Eval: {eval_loss}')

# Compute Metrics
print(f'\n===== Metrics =====')
y_pred, y_true = utils.predict(model, test_loader, device, True)
_logger.debug(y_true.shape)
_logger.debug(y_pred.shape)

## Metrics to compute
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = root_mean_squared_error (y_true, y_pred)

# Log on file
with open(LOG_FILENAME, 'a') as log:
    line = f'AutoEncoder,{FILTERS},{EMBEDDING_CHANNELS},{total_params},{model.compression_ratio}:1,{SEED},{INPUTS_SHAPE[1]},{INPUTS_SHAPE[0]},{BATCH_SIZE},{EPOCHS},{mae},{mse},{rmse}\n'
    log.write(line)
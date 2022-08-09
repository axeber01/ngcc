import numpy as np

# Training room simulation parameters
# room dimensions in meters
dx_train = 7.0
dy_train = 5.0
dz_train = 3.0
room_dim_train = [dx_train, dy_train, dz_train]
xyz_min_train = [0.0, 0.0, 0.0]
xyz_max_train = room_dim_train

# microphone locations
mic_locs_train = np.array([[3.5, 2.25, 1.5], [3.5, 2.75, 1.5]]).T

# Test room parameters
dx_test = 6.0
dy_test = 4.0
dz_test = 2.5
room_dim_test = [dx_test, dy_test, dz_test]
xyz_min_test = [0.0, 0.0, 0.0]
xyz_max_test = room_dim_test

mic_locs_test = np.array([[3.0, 1.75, 1.25], [3.0, 2.25, 1.25]]).T

# Testing environment configuration
# the model will be evaluated for all SNRs and T60s in the lists
snr_range = [0, 6, 12, 18, 24, 30]
t60_range = [0.2, 0.4, 0.6]

# accuracy threshold in cm
t_cm = 10

# Training hyperparams
seed = 0
batch_size = 32
epochs = 30
lr = 0.001  # learning rate
wd = 0.0  # weight decay
ls = 0.0  # label smoothing

# Model parameters
model = 'NGCCPHAT'  # choices: NGCCPHAT, PGCCPHAT
max_delay = 23
num_channels = 128  # number of channels in final layer of NGCCPHAT backbone
head = 'classifier'  # final layer type. Choices: 'classifier', 'regression'
loss = 'ce'  # use 'ce' loss for classifier and 'mse' loss for regression
# Set to true in order to replace Sinc filters with regular convolutional layers
no_sinc = False

# training environment
snr = [0, 30]  # during training, snr will be drawn uniformly from this interval
t60 = [0.2, 1.0]  # during training, t60 will be drawn uniformly from this interval
fs = 16000  # sampling rate
sig_len = 2048  # length of snippet used for tdoa estimation
anechoic = False  # set to True to use anechoic environment without reverberation

# threshold in samples
t = t_cm * fs / (343 * 100)

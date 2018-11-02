from datageneration.datagenerator import MatrixProjection
from datageneration.datagenerator import RandomMatrixProjection
from datageneration.datagenerator import SubsetsLatentZ3

from models.autoencoder import Autoencoder
from models.autoencoder import MultiAutoencoder
from utils.time import timeSince
from utils.launcher import launcher_predict

import torch.nn as nn
import torch.optim as optim

import numpy as np
import time
import torch
import math


#### Data Generation

N = 3
dim_z = 3
dim_input = 2
subset3 = SubsetsLatentZ3(noise_variance=1e-2)
z = np.random.normal(size=(dim_z, 1))
samples = subset3.sample(z, noise=True)





#%%# launch

# Create the NN
encoder_layer_1_size = 2
encoder_layer_2_size = 2
latent_layer_size = 3
decoder_layer_1_size = 2
decoder_layer_2_size = 2

multiencoder = MultiAutoencoder(dim_input, encoder_layer_1_size, encoder_layer_2_size,
                                latent_layer_size,
                            decoder_layer_1_size, decoder_layer_2_size,
                            N)

## Launch parameters
lr = 0.001
criterion = nn.MSELoss()
optimizer = optim.Adam(multiencoder.parameters(), lr=lr)

batch_size = int(1e1)
n_iter = int(1e3)
n_epoch = 10

start = time.time()


for epoch in range(n_epoch):
    for i in range(n_iter):

        z_in = np.random.normal(size=(dim_z, 1))
        multi_x = subset3.sample(z_in, noise=True)

        # Randomly remove modalities

        # Initialize hidden, grad, loss
        multiencoder.zero_grad()
        loss = 0

        output = multiencoder(multi_x)
        loss = criterion(torch.cat(output).view(N, 1, dim_input), torch.cat(multi_x)) # Not sure here with the dimensions
        #loss = criterion(output, torch.FloatTensor(z_in).view(1, -1)) # Not sure here with the dimensions

        # Gradient step
        loss.backward(retain_graph=True)
        optimizer.step()

    print("Epoch {0}, {1} : {2}".format(epoch, timeSince(start), loss.data.view(-1).numpy()[0]))


#%% Learned Representation

z_in = np.random.normal(size=(dim_z, 1))
multi_x = subset3.sample(z_in, noise=True)
z_latent = multiencoder(multi_x)
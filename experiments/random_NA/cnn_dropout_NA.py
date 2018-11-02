#%%
import numpy as np
import sys
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from datageneration.cognitive_scores import CognitiveScores

import time
from utils.time import timeSince

#%% Experiment parameters
dim_time = 12
dim_input = 5

generator = CognitiveScores(dim_time, dim_input)

#%%
# Model parameters

sample = generator.sample()
input = torch.FloatTensor(sample.values[:,2:]).view(1, dim_time, dim_input)
plt.plot(input.data.numpy()[0,:,:])
plt.show()


#%% CNN Autoencoder

class Autoencoder_CNN(nn.Module):

    def __init__(self, batch_size, dim_time, dim_input):
        super(Autoencoder_CNN, self).__init__()

        self.batch_size = batch_size
        self.dim_time = dim_time
        self.dim_input = dim_input

        # Encoder
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(1, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=(1, 2), stride=2, padding=1)
        self.linear1 = nn.Linear(2*8*3, 8)

        # Decoder
        self.linear2 = nn.Linear(8, 12)
        self.linear3 = nn.Linear(12, 16)
        self.linear4 = nn.Linear(16, self.dim_time*self.dim_input)

        # Dropout
        self.dropout = nn.Dropout(p=0.2)


    def encoder(self, x):
        x = self.dropout(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.linear1(x.view(self.batch_size, 1, 2*8*3))
        return x

    def decoder(self, x):
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        x = x.view(self.batch_size, 1, self.dim_time, self.dim_input)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



autoencoder_cnn = Autoencoder_CNN(1, dim_time, dim_input)

dim_time = 12
dim_input = 5
sample = generator.sample()
input = torch.FloatTensor(sample.values[:,2:]).view(1,1, dim_time, dim_input) # batch / channels / dim 1 / dim 2
res = autoencoder_cnn(input)



#%% Launch experiment

batch_size = 20
autoencoder_cnn = Autoencoder_CNN(batch_size, dim_time, dim_input)


## Launch parameters
lr = 0.0001
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder_cnn.parameters())


n_iter = int(1e2)
n_epoch = 10

start = time.time()


for epoch in range(n_epoch):
    for i in range(n_iter):

        # Generate inputs
        samples = [generator.sample() for j in range(batch_size)]
        inputs = [torch.FloatTensor(sample.values[:, 2:]).view(1, 1, dim_time, dim_input) for sample in samples]
        input = torch.cat(inputs, dim=0)


        # Initialize hidden, grad, loss
        autoencoder_cnn.zero_grad()
        loss = 0

        output = autoencoder_cnn(input)
        loss = criterion( output, input) # Not sure here with the dimensions

        # Gradient step
        loss.backward()
        optimizer.step()

    print("Epoch {0}, {1} : {2}".format(epoch, timeSince(start), loss.data.view(-1).numpy()[0]))


#%% Visualization of results

i = np.random.randint(0,1000)

# Generate inputs
samples = [generator.sample() for j in range(batch_size)]
inputs = [torch.FloatTensor(sample.values[:, 2:]).view(1, 1, dim_time, dim_input) for sample in samples]
input = torch.cat(inputs, dim=0)
output = autoencoder_cnn(input)

colors = ['blue','red','darkgreen','purple']

for dim in range(dim_input):
    plt.plot(input.data.numpy()[0, 0, :, dim], c=colors[dim])
    plt.plot(output.data.numpy()[0, 0, :, dim], linestyle='--', c = colors[dim])
plt.ylim(0,1)
plt.show()


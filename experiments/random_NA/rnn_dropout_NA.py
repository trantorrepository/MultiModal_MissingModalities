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


#%% RNN

# input_size / hidden_size / num_layers / nonlinearity / batch_first

input_size = dim_input
batch_size = 3

hidden_size = 10

input = torch.randn(batch_size, dim_time, dim_input) # batch / length / dim
h0 = torch.zeros(2, batch_size, hidden_size) # num_layers / batch / dim




class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 batch_size, time_steps):
        super(RNN, self).__init__()

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.input_size = input_size

        self.rnn = nn.RNN(input_size, hidden_size, 2, batch_first=True)
        self.o2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.o2o(output[:,-1,:])
        output = output.view(self.batch_size, self.time_steps,  self.input_size)
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

rnn = RNN(input_size, hidden_size, dim_input*dim_time, batch_size, dim_time)
output = rnn(input, h0)


#%% Launch

batch_size = 20
rnn = RNN(input_size, hidden_size, dim_input*dim_time, batch_size, dim_time)


## Launch parameters
lr = 0.0001
criterion = nn.MSELoss()
optimizer = optim.Adam(rnn.parameters())


n_iter = int(1e2)
n_epoch = 10

start = time.time()


for epoch in range(n_epoch):
    for i in range(n_iter):

        # Generate inputs
        samples = [generator.sample() for j in range(batch_size)]
        inputs = [torch.FloatTensor(sample.values[:, 2:]).view(1, dim_time, dim_input) for sample in samples]
        input = torch.cat(inputs, dim=0)


        # Initialize hidden, grad, loss
        rnn.zero_grad()
        loss = 0

        h0 = torch.zeros(2, batch_size, hidden_size)  # num_layers / batch / dim
        output = rnn(input, h0)
        loss = criterion( output, input) # Not sure here with the dimensions

        # Gradient step
        loss.backward()
        optimizer.step()

    print("Epoch {0}, {1} : {2}".format(epoch, timeSince(start), loss.data.view(-1).numpy()[0]))


#%% Visualization of results

i = np.random.randint(0,1000)

# Generate inputs
samples = [generator.sample() for j in range(batch_size)]
inputs = [torch.FloatTensor(sample.values[:, 2:]).view(1, dim_time, dim_input) for sample in samples]
input = torch.cat(inputs, dim=0)
h0 = torch.zeros(2, batch_size, hidden_size)  # num_layers / batch / dim
output = rnn(input, h0)

colors = ['blue','red','darkgreen','purple', 'brown']

for dim in range(dim_input):
    plt.plot(input.data.numpy()[0, :, dim], c=colors[dim])
    plt.plot(output.data.numpy()[0, :, dim], linestyle='--', c = colors[dim])
plt.ylim(0,1)
plt.show()


"""
Toy 1 is a linear transformation of an input of given size.
Task is easy for a nn
"""



from utils.time import timeSince


import torch.nn as nn
import torch.optim as optim

import numpy as np
import time
import torch


#### Data Generation

dim_input = 1
batch_size = 20
z = np.random.normal(size=(batch_size, dim_input))
sample = torch.FloatTensor(2*z+1+np.random.normal(0, 1e-2, size=(batch_size, dim_input))).view(batch_size, 1, dim_input)


#%% Multi autoencoder

import torch.nn.functional as F

## Multi Autoencoder
class Net(nn.Module):

    def __init__(self, input_size, latent_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, latent_size)

    def forward(self, x):
        x = self.fc1(x)
        return x


net = Net(dim_input, dim_input)
out = net(sample)

#%%# launch

# Create the NN

net = Net(dim_input, dim_input)

## Launch parameters
lr = 0.005
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

batch_size = int(1e1)
n_iter = int(1e3)
n_epoch = 10

start = time.time()


for epoch in range(n_epoch):
    for i in range(n_iter):


        z = np.random.normal(size=(batch_size, dim_input))
        sample = torch.FloatTensor(2*z + 10 + np.random.normal(0, 1e-2,
                                                            size=(batch_size, dim_input))).view(batch_size, 1,
                                                                                                         dim_input)

        # Randomly remove modalities

        # Initialize hidden, grad, loss
        net.zero_grad()
        loss = 0

        output = net(sample)
        loss = criterion(output, torch.FloatTensor(z).view(batch_size, 1, dim_input)) # Not sure here with the dimensions
        #loss = criterion(output, torch.FloatTensor(z_in).view(1, -1)) # Not sure here with the dimensions

        # Gradient step
        loss.backward()
        optimizer.step()

    print("Epoch {0}, {1} : {2}".format(epoch, timeSince(start), loss.data.view(-1).numpy()[0]))


#%% Learned Representation

z = np.random.normal(size=(batch_size, dim_input))
sample = torch.FloatTensor(2*z + 10 + np.random.normal(0, 1e-2,
                                                    size=(batch_size, dim_input))).view(batch_size, 1,
                                                                                        dim_input)
out = net(sample)

print(sample, out)
print(list(net.parameters()))
"""
toy5 expands toy3, with the adding of linear combinations.
Inputs are supposed to represent different aspects of the same input data, by analogy with combining different
neuro-imaging modalities.

0: x
1: y
2: x+y
3: x-y
4: xy
5: x^2
6: y^2

Dim 0 : batch
Dim 1 : modalities when aggregated
DIm 2 : dimension z

"""



from utils.time import timeSince


import torch.nn as nn
import torch.optim as optim

import numpy as np
import time
import torch
import torch.nn.functional as F
from datageneration.datagenerator import Toy3_nonlinear


#### Data Generation

dim_input = 1
batch_size = 10

generator = Toy3_nonlinear(dim_input=dim_input, noise_variance=1e-2)

(x,y), sample = generator.sample( batch_size=batch_size)


#%%  ## Multi Autoencoder

class Encoder(nn.Module):

    def __init__(self, input_size, latent_size):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_size, latent_size)
        self.fc2 = nn.Linear(latent_size, latent_size)
        self.fc3 = nn.Linear(latent_size, latent_size)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        return x

class MultiAutoencoder(nn.Module):

    def __init__(self, input_size, latent_layer_size, output_size , N, batch_size=1):
        super(MultiAutoencoder, self).__init__()

        self.N = N
        self.encoders = [Encoder(input_size,
                                 latent_layer_size) for i in range(N)]
        self.decoders = [Decoder(latent_layer_size,
                                 input_size) for i in range(N)]

        self.fc1 = nn.Linear(latent_layer_size, latent_layer_size)
        self.fc2 = nn.Linear(latent_layer_size, latent_layer_size)
        self.fc3 = nn.Linear(latent_layer_size, output_size)

        self.dim_latent = latent_layer_size
        self.batch_size = batch_size

    def encoder(self, multi_x, indices):
        x = [self.encoders[i](multi_x[i]) for i in indices]
        return x

    #@staticmethod
    def fusion(self, multi_x):
        x = torch.cat(multi_x, dim=1)
        x = x.max(dim=1)[0].view(self.batch_size, 1, self.dim_latent)
        #x = x[:, 1, :].view(batch_size, 1, self.dim_latent)
        #x = x.mean(dim=1).view(self.batch_size, 1, self.dim_latent)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def latent(self, multi_x, indices):
        x = self.encoder(multi_x, indices)
        x = self.fusion(x)
        return x

    def decoder(self, latent_x):
        multi_x = [self.decoders[i](latent_x) for i in range(self.N)]
        return multi_x

    def forward(self, x, indices):
        x = self.encoder(x, indices)
        x = self.fusion(x)
        x = self.decoder(x)
        return x

    def zero_grad(self):
        for encoder in self.encoders:
            encoder.zero_grad()
        for decoder in self.decoders:
            decoder.zero_grad()


batch_size = 1
output_size = 2
indices = [0, 1, 2]
dim_latent = 2
multi_autoencoder = MultiAutoencoder(dim_input, dim_latent, output_size, 3, batch_size)
pre_fusion = multi_autoencoder.encoder(sample, indices)


res = multi_autoencoder.latent(sample, indices)

#%% Launch NN


# Create the NN
dim_input = 1
dim_latent = 4
dim_output = 2
batch_size = int(1e2)
N = 3

multi_autoencoder = MultiAutoencoder(dim_input, dim_latent, output_size, N, batch_size)

parameters_0 = list(multi_autoencoder.encoders[0].parameters())[0].data.numpy().copy()


## Launch parameters
lr = 0.0005
criterion = nn.MSELoss()
optimizers = [optim.Adam(encoder.parameters(), lr=lr) for encoder in multi_autoencoder.encoders]
optimizers.append(optim.Adam(multi_autoencoder.parameters(), lr=lr))

n_iter = int(2e3)
n_epoch = 10

start = time.time()


for epoch in range(n_epoch):
    for i in range(n_iter):

        z = np.random.normal(size=(batch_size, dim_input))
        (x, y), sample = generator.sample(batch_size=batch_size)

        indices = np.unique(np.random.choice((0,1,2), 3))
        #indices = [0,1,2]

        # Randomly remove modalities

        # Initialize hidden, grad, loss
        multi_autoencoder.zero_grad()
        loss = 0

        output = multi_autoencoder.latent(sample, indices)
        loss = criterion(output,
                          torch.cat((torch.FloatTensor(x), torch.FloatTensor(y)), dim=1).view(batch_size, 1, dim_output)) # Not sure here with the dimensions


        # Gradient step
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

    print("Epoch {0}, {1} : {2}".format(epoch, timeSince(start), loss.data.view(-1).numpy()[0]))

parameters_0_end = list(multi_autoencoder.encoders[0].parameters())[0].data.numpy()



#print(parameters_0)
#print(parameters_0_end)


#%% Test Error

test_size = int(1e3)
multi_autoencoder.batch_size = test_size
dim_output = 2

indices = [2]
(x, y), sample = generator.sample(batch_size=test_size)


output = multi_autoencoder.latent(sample, indices)

difference = criterion(output,
                          torch.cat((torch.FloatTensor(x), torch.FloatTensor(y)), dim=1).view(test_size, 1, dim_output))
print("Test error : {0}".format(difference))


#%% Visualization
import matplotlib.pyplot as plt

batch_size = 100
multi_autoencoder.batch_size = batch_size

indices = [0]

grid = np.linspace(-1, 1, batch_size)

x = grid.reshape(batch_size, dim_input)
y = np.zeros(shape=(batch_size, dim_input))

#x,y = y,x


(x,y), sample = generator.sample(z=(x,y), batch_size=batch_size)
output = multi_autoencoder.latent(sample, indices)
output = output.view(batch_size, 2).data.numpy()

plt.plot(grid, output)
plt.legend(['x','y'])
plt.show()

"""
Toy 3 expandas toy 1 example, with a list of 3 intputs instead of one.
Inputs are supposed to represent different aspects of the same input data, by analogy with combining different
neuro-imaging modalities.

We choose 2 linear modalities, and a third that does not add information (here noise).

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

np.random.seed(0)
torch.manual_seed(0)

#### Data Generation

dim_input = 1
batch_size = 10
z = np.random.normal(size=(batch_size, dim_input))
sample = [torch.FloatTensor(-z+1+np.random.normal(0, 1e-2,
                                                  size=(batch_size, dim_input))).view(batch_size, 1, dim_input),
torch.FloatTensor(3*z-2+np.random.normal(0, 1e-2,
                                                  size=(batch_size, dim_input))).view(batch_size, 1, dim_input),
torch.FloatTensor(np.random.normal(0, 1e-2,
                                                  size=(batch_size, dim_input))).view(batch_size, 1, dim_input)]





#%%  ## Multi Autoencoder

class Encoder(nn.Module):

    def __init__(self, input_size, latent_size):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_size, latent_size)

    def forward(self, x):
        x = self.fc1(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        return x

class MultiAutoencoder():
    def __init__(self, input_size, latent_layer_size, N, batch_size=1):

        self.N = N
        self.encoders = [Encoder(input_size,
                                 latent_layer_size) for i in range(N)]
        self.decoders = [Decoder(latent_layer_size,
                                 input_size) for i in range(N)]

        #self.dropout = nn.Dropout(0.25)

        self.dim_latent = latent_layer_size
        self.batch_size = batch_size

    def encoder(self, multi_x, indices):
        x = [self.encoders[i](multi_x[i]) for i in indices]
        return x

    #@staticmethod
    def fusion(self, multi_x):
        x = torch.cat(multi_x, dim=1)
        #x = x.max(dim=0)[0]
        #x = x[:, 1, :].view(batch_size, 1, self.dim_latent)
        x = x.mean(dim=1).view(self.batch_size, 1, self.dim_latent)
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

#[idx for idx, a in enumerate(sample) if not(np.isnan(a).data[0,0,0].numpy())]

indices = (0,1,2)
dim_latent = 1
multi_autoencoder = MultiAutoencoder(dim_input, dim_latent, 3, batch_size)
pre_fusion = multi_autoencoder.encoder(sample, indices)

x = torch.cat(pre_fusion, dim=1)
print(x.size())

res = multi_autoencoder.latent(sample, indices)

#%% Launch NN


# Create the NN
dim_input = 1
dim_latent = 1
batch_size = int(2e1)

multi_autoencoder = MultiAutoencoder(dim_input, dim_latent, 3, batch_size)

parameters_0 = list(multi_autoencoder.encoders[0].parameters())[0].data.numpy().copy()


## Launch parameters
lr = 0.0005
criterion = nn.MSELoss()
optimizers = [optim.Adam(encoder.parameters(), lr=lr) for encoder in multi_autoencoder.encoders]


n_iter = int(1e3)
n_epoch = 15

start = time.time()


for epoch in range(n_epoch):
    for i in range(n_iter):

        z = np.random.normal(size=(batch_size, dim_input))
        sample = [torch.FloatTensor(-z + 1 + np.random.normal(0, 1e-2,
                                                              size=(batch_size, dim_input))).view(batch_size, 1,
                                                                                                  dim_input),
                  torch.FloatTensor(3 * z - 2 + np.random.normal(0, 1e-2,
                                                                 size=(batch_size, dim_input))).view(batch_size, 1,
                                                                                                     dim_input),
                  torch.FloatTensor(np.random.normal(1, 1e0,
                                                     size=(batch_size, dim_input))).view(batch_size, 1, dim_input)]

        indices = np.unique(np.random.choice((0,1,2),3))
        indices = [0,1,2]

        # Randomly remove modalities

        # Initialize hidden, grad, loss
        multi_autoencoder.zero_grad()
        loss = 0

        output = multi_autoencoder.latent(sample, indices)
        loss = criterion(output, torch.FloatTensor(z).view(batch_size, 1, dim_input)) # Not sure here with the dimensions
        #loss = criterion(output, torch.FloatTensor(z_in).view(1, -1)) # Not sure here with the dimensions

        # Gradient step
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

    print("Epoch {0}, {1} : {2}".format(epoch, timeSince(start), loss.data.view(-1).numpy()[0]))

parameters_0_end = list(multi_autoencoder.encoders[0].parameters())[0].data.numpy()

#%%

print(parameters_0)
print(parameters_0_end)


#%%

test_size = int(1e5)
multi_autoencoder.batch_size = test_size

indices = [0,1,2]
z = np.random.normal(size=(test_size, dim_input))
sample = [torch.FloatTensor(-z + 1 + np.random.normal(0, 1e-2,
                                                      size=(test_size, dim_input))).view(test_size, 1,
                                                                                          dim_input),
          torch.FloatTensor(3 * z - 2 + np.random.normal(0, 1e-2,
                                                         size=(test_size, dim_input))).view(test_size, 1,
                                                                                             dim_input),
          torch.FloatTensor(np.random.normal(1, 1e-2,
                                             size=(test_size, dim_input))).view(test_size, 1, dim_input)]


output = multi_autoencoder.latent(sample, indices)

difference = z-output.data.numpy().reshape(test_size, 1)
print("Test error : {0}".format(np.linalg.norm(difference)/difference.shape[0]))
#print(list(multi_autoencoder.encoders[0].parameters()))
#print(list(multi_autoencoder.encoders[1].parameters()))
#print(list(multi_autoencoder.encoders[2].parameters()))
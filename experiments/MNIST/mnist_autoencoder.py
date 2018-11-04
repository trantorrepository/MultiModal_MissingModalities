import torch

import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import torch.optim as optim
import time
import math
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.nn.functional as F
from utils.time import timeSince

torch.manual_seed(0)

#%%


def multimodal_transformation(input):

    modality_0 = torchvision.transforms.functional.rotate(input, -45)
    modality_0 = torchvision.transforms.functional.crop(modality_0, 0, 12, 28, 28)

    modality_1 = torchvision.transforms.functional.crop(input, 0, 0, 28, 28)

    modalities = [modality_0, modality_1]
    modalities = [torchvision.transforms.functional.to_tensor(modality) for modality in modalities]
    modalities = [torch.autograd.Variable(modality) for modality in modalities]
    return modalities


# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 10
DOWNLOAD_MNIST = True


train_data_transform = torchvision.datasets.MNIST(
    root='./data/mnist/',
    train=True,                                     # this is training data
    transform=multimodal_transformation,    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,                        # download it if you don't have it
)






# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data_transform, batch_size=BATCH_SIZE, shuffle=True)





#%% Multi-CNN architecture


class Image_Encoder(nn.Module):
    def __init__(self, dim_latent):
        super(Image_Encoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(12, 8, kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc = nn.Linear(32, dim_latent)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # torch.Size([32, 12, 13, 13])
        x = self.maxpool1(x)  # torch.Size([32, 12, 6, 6])
        x = F.relu(self.conv2(x))  # torch.Size([32, 8, 3, 3])
        x = self.maxpool2(x)  # torch.Size([32, 8, 2, 2])
        x = self.fc(x.view(-1,1,32))
        return x


class Image_Decoder(nn.Module):

    def __init__(self, dim_latent):
        super(Image_Decoder, self).__init__()

        # Encoder
        self.fc = nn.Linear(dim_latent,2*2*8)
        self.deconv3 = nn.ConvTranspose2d(in_channels=8,out_channels=16, kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=16,out_channels=8, kernel_size=5,stride=3, padding = 1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=8,out_channels=1, kernel_size=2, stride=2, padding = 1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1,8,2,2)
        x = F.relu(self.deconv3(x)) # torch.Size([32, 9, 5, 5])
        x = F.relu(self.deconv2(x)) # torch.Size([32, 12, 14, 14])
        x = F.sigmoid(self.deconv1(x)) # torch.Size([32, 1, 28, 28])
        return x




class Autoencoder(nn.Module):

    def __init__(self, encoder, decoder,  dim_latent, batch_size=1):
        super(Autoencoder, self).__init__()

        # Encoders/decoders
        self.encoder = encoder
        self.decoder = decoder

        # Utils
        self.batch_size = batch_size
        self.dim_latent = dim_latent

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def zero_grad(self):
            self.encoder.zero_grad()
            self.decoder.zero_grad()







dim_latent = 10
encoder = Image_Encoder(dim_latent)
decoder = Image_Decoder(dim_latent)
multi_autoencoder = Autoencoder(encoder, decoder,  dim_latent, batch_size=BATCH_SIZE)





#%% Train the CNN



def plot_reconstruction(multi_autoencoder, modality, epoch=0, batch_limit=1, plot=False):

    output = multi_autoencoder(modality)

    batch_plot = min(batch_limit, multi_autoencoder.batch_size)

    fig, axes = plt.subplots(batch_limit,2, figsize=(10,batch_plot*5))

    for batch in range(batch_limit):

            axes[batch, 0].imshow(modality[batch,0,:,:].data.cpu().numpy())
            axes[batch, 1].imshow(output[batch,0, :, :].data.cpu().numpy())


    plt.tight_layout()
    plt.savefig('./experiments/MNIST/output/basic_mnist_epoch{0}.pdf'.format(epoch))

    if plot:
        plt.show()



# Create the NN
use_gpu = False
batch_size = BATCH_SIZE
dim_latent = 10
encoder = Image_Encoder(dim_latent)
decoder = Image_Decoder(dim_latent)
multi_autoencoder = Autoencoder(encoder, decoder,  dim_latent, batch_size=BATCH_SIZE)


## Launch parameters
lr = 0.005
criterion_img = nn.MSELoss()
criterion_number = nn.CrossEntropyLoss()
optimizer = optim.Adam(multi_autoencoder.parameters(), lr=lr)

n_epoch = 10

start = time.time()

if use_gpu:
    multi_autoencoder=multi_autoencoder.cuda()

for epoch in range(n_epoch):
    for i, (img_modalities, label) in enumerate(train_loader):

        modality = img_modalities[1]

        if use_gpu:
            modality=modality.cuda()


        # Randomly remove modalities

        # Initialize hidden, grad, loss
        multi_autoencoder.zero_grad()
        loss = 0

        output = multi_autoencoder(modality)
        loss = criterion_img(output, modality)


        # Gradient step
        loss.backward()
        optimizer.step()

        if i>50:
            break

    plot_reconstruction(multi_autoencoder, modality, epoch=epoch, batch_limit=8, plot=False)
    print("Epoch {0}, {1} : {2}".format(epoch, timeSince(start), loss.data.view(-1).cpu().numpy()[0]))


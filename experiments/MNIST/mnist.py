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

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005         # learning rate
DOWNLOAD_MNIST = True
N_TEST_IMG = 5

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./data/mnist/',
    train=True,                                     # this is training data
    #transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,                        # download it if you don't have it
)

# plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)



#%% Transformations used

def multimodal_transformation(input):

    modality_0 = torchvision.transforms.functional.rotate(input, -45)
    modality_0 = torchvision.transforms.functional.crop(modality_0, 0, 12, 28, 28)

    modality_1 = torchvision.transforms.functional.crop(input, 8, 0, 28, 28)

    modalities = [modality_0, modality_1]
    modalities = [torchvision.transforms.functional.to_tensor(modality) for modality in modalities]
    #modalities = [torch.autograd.Variable(modality, requires_grad=True) for modality in modalities]
    return modalities

idx = 0
input = train_data[idx]
modalities = multimodal_transformation(input[0])

for modality in modalities:
    plt.imshow(modality[0,:,:].data.numpy())
    plt.show()


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
        self.fc = nn.Linear(dim_latent, 32)
        self.deconv3 = nn.ConvTranspose2d(in_channels=8,out_channels=9,kernel_size=3,stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=9,out_channels=12,kernel_size=5,stride=3, padding = 1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=12,out_channels=1,kernel_size=2,stride=2, padding = 1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1,8,2,2)
        x = F.relu(self.deconv3(x)) # torch.Size([32, 9, 5, 5])
        x = F.relu(self.deconv2(x)) # torch.Size([32, 12, 14, 14])
        x = F.sigmoid(self.deconv1(x)) # torch.Size([32, 1, 28, 28])
        return x



class Autoencoder(nn.Module):

    def __init__(self, encoders, decoders,  dim_latent, batch_size=1):
        super(Autoencoder, self).__init__()

        # Encoders/decoders
        self.encoders = encoders
        self.decoders = decoders

        # Utils
        self.batch_size = batch_size
        self.dim_latent = dim_latent
        self.N = len(encoders)

    def encoder(self, multi_x, indices):
        x = [self.encoders[i](multi_x[i]) for i in indices]
        return x

    def fusion(self, multi_x):
        x = torch.cat(multi_x, dim=1)
        #x = x.mean(dim=1).view(self.batch_size, 1, self.dim_latent)
        x = x.max(dim=1)[0].view(self.batch_size, 1, self.dim_latent)
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

dim_latent = 10
encoder_1 = Image_Encoder(dim_latent)
encoder_2 = Image_Encoder(dim_latent)
decoder_1 = Image_Decoder(dim_latent)
decoder_2 = Image_Decoder(dim_latent)

encoders = [encoder_1, encoder_2]
decoders = [decoder_1, decoder_2]


multi_autoencoder = Autoencoder( encoders, decoders,  dim_latent)

indices = [0,1]
multi_autoencoder.forward([modalities[0].view(1,1,28,28), modalities[0].view(1,1,28,28)], indices)



#%% Train the CNN



# Create the NN
dim_latent = 10
batch_size = int(1e2)

# Create the Auto encoder
encoder_1 = Image_Encoder(dim_latent)
encoder_2 = Image_Encoder(dim_latent)
decoder_1 = Image_Decoder(dim_latent)
decoder_2 = Image_Decoder(dim_latent)
encoders = [encoder_1, encoder_2]
decoders = [decoder_1, decoder_2]
multi_autoencoder = Autoencoder(encoders, decoders,  dim_latent)


## Launch parameters
lr = 0.0005
criterion = nn.MSELoss()
optimizers_encoders = [optim.Adam(encoder.parameters(), lr=lr) for encoder in multi_autoencoder.encoders]
optimizers_decoders = [optim.Adam(decoder.parameters(), lr=lr) for decoder in multi_autoencoder.decoders]
optimizers = optimizers_encoders+optimizers_decoders


n_iter = int(2e3)
n_epoch = 10

start = time.time()


for epoch in range(n_epoch):
    for i in range(n_iter):

        input = train_data[i]
        modalities = multimodal_transformation(input[0])
        modalities = [modality.view(1, 1, 28, 28) for modality in modalities]

        #indices = np.unique(np.random.choice((0,1,2), 3))
        indices = [0, 1]

        # Randomly remove modalities

        # Initialize hidden, grad, loss
        multi_autoencoder.zero_grad()
        loss = 0

        output = multi_autoencoder(modalities, indices)
        loss = criterion(output[0], modalities[0])+criterion(output[1], modalities[1])

        # Gradient step
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

    print("Epoch {0}, {1} : {2}".format(epoch, timeSince(start), loss.data.view(-1).numpy()[0]))
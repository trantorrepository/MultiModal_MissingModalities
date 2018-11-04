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

    modality_1 = torchvision.transforms.functional.crop(input, 8, 0, 28, 28)

    modalities = [modality_0, modality_1]
    modalities = [torchvision.transforms.functional.to_tensor(modality) for modality in modalities]
    modalities = [torch.autograd.Variable(modality) for modality in modalities]
    return modalities


# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 20
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
        self.fc = nn.Linear(dim_latent,4*4*8)
        self.deconv3 = nn.ConvTranspose2d(in_channels=8,out_channels=32,kernel_size=2,stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=4,stride=2, padding = 1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=16,out_channels=1, kernel_size=2, stride=2, padding = 2)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1,8,4,4)
        x = F.relu(self.deconv3(x)) # torch.Size([32, 9, 5, 5])
        x = F.relu(self.deconv2(x)) # torch.Size([32, 12, 14, 14])
        x = F.sigmoid(self.deconv1(x)) # torch.Size([32, 1, 28, 28])
        return x


class Number_Encoder(nn.Module):
    def __init__(self, dim_input ,dim_latent):
        super(Number_Encoder, self).__init__()
        self.fc = nn.Linear(dim_input, dim_latent)

    def forward(self, x):
        x = self.fc(x)
        return x


class Number_Decoder(nn.Module):
    def __init__(self, dim_input,dim_latent):
        super(Number_Decoder, self).__init__()
        self.fc = nn.Linear(dim_latent, dim_input)

    def forward(self, x):
        x = self.fc(x)
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






dim_input = 10
dim_latent = 10
encoder_1 = Image_Encoder(dim_latent)
encoder_2 = Image_Encoder(dim_latent)
encoder_3 = Number_Encoder(dim_input, dim_latent)

decoder_1 = Image_Decoder(dim_latent)
decoder_2 = Image_Decoder(dim_latent)
decoder_3 = Number_Decoder(dim_input, dim_latent)

encoders = [encoder_1, encoder_2, encoder_3]
decoders = [decoder_1, decoder_2, decoder_3]


multi_autoencoder = Autoencoder(encoders, decoders,  dim_latent, batch_size=BATCH_SIZE)

indices = [0,1]


# Sample


for i, (modalities, label) in enumerate(train_loader):
    print(modalities)
    if i>1:
        break


for modality in modalities:
    plt.imshow(modality[0,0,:,:].data.numpy())
    plt.show()

#%%

batch_size = BATCH_SIZE
label_onehot = torch.zeros(batch_size, dim_input)
label_onehot = label_onehot.scatter(1, label.view(-1,1),1)


multi_autoencoder.forward([modalities[0],
                           modalities[1],
                           label_onehot],
                          indices)




#%% Train the CNN



def plot_reconstruction(multi_autoencoder, sample,
                        indices, batch_size, epoch=0, batch_limit=1, plot=False):

    # Process input data
    (img_modalities, label) = sample

    label_onehot = torch.zeros(batch_size, 10)
    label_onehot = label_onehot.scatter(1, label.view(-1, 1), 1)

    modalities = img_modalities + [torch.FloatTensor(label_onehot.numpy()).view(batch_size, 1, dim_input)]
    output = multi_autoencoder(modalities, indices)

    batch_plot = min(batch_limit, multi_autoencoder.batch_size)

    fig, axes = plt.subplots(2*batch_limit,3, figsize=(10,batch_plot*5))

    for batch in range(batch_limit):

        for j, modality in enumerate([modalities[idx] for idx in [0,1]]):
            axes[2*batch, j].imshow(modality[batch,0,:,:].data.numpy())
            if j>1:
                break

        for j, output_modality in enumerate([output[idx] for idx in [0,1]]):
            axes[2*batch+1, j].imshow(output_modality[batch,0, :, :].data.numpy())
            if j>1:
                break

        axes[2*batch, 2].bar(range(0, 10),label_onehot[batch, :].data.numpy())
        axes[2*batch+1, 2].bar(range(0, 10),output[2][batch, 0, :].data.numpy())

    plt.tight_layout()
    plt.savefig('/Users/raphael.couronne/Programming/ARAMIS/Projects/MultiModal_MissingModalities/experiments/MNIST/output/mnist_epoch{0}.pdf'.format(epoch))
    if plot:
        plt.show()


# Create the NN
batch_size = BATCH_SIZE
dim_latent = 14

# Create the Auto encoder
encoder_1 = Image_Encoder(dim_latent)
encoder_2 = Image_Encoder(dim_latent)
encoder_3 = Number_Encoder(dim_input, dim_latent)

decoder_1 = Image_Decoder(dim_latent)
decoder_2 = Image_Decoder(dim_latent)
decoder_3 = Number_Decoder(dim_input, dim_latent)

encoders = [encoder_1, encoder_2, encoder_3]
decoders = [decoder_1, decoder_2, decoder_3]

multi_autoencoder = Autoencoder(encoders, decoders,  dim_latent,  batch_size)


## Launch parameters
lr = 0.0005
criterion_img = nn.MSELoss()
criterion_number = nn.CrossEntropyLoss()
optimizers_encoders = [optim.Adam(encoder.parameters(), lr=lr) for encoder in multi_autoencoder.encoders]
optimizers_decoders = [optim.Adam(decoder.parameters(), lr=lr) for decoder in multi_autoencoder.decoders]
optimizers = optimizers_encoders+optimizers_decoders

n_epoch = 10

start = time.time()


for epoch in range(n_epoch):
    for i, (img_modalities, label) in enumerate(train_loader):

        # Add the label
        label_onehot = torch.zeros(batch_size, dim_input)
        label_onehot = label_onehot.scatter(1, label.view(-1, 1), 1)

        modalities = img_modalities+[torch.FloatTensor(label_onehot).view(batch_size, 1, dim_input)]

        indices = [0,1]
        #indices = np.unique(np.random.choice((0,1, 2), 3))

        # Randomly remove modalities

        # Initialize hidden, grad, loss
        multi_autoencoder.zero_grad()
        loss = 0

        output = multi_autoencoder(modalities, indices)
        losses = [criterion_img(output[0], modalities[0]),
               criterion_img(output[1], modalities[1]),
               criterion_number(output[2].view(batch_size, dim_input), label)]

        for idx in indices:
            loss+=losses[idx]


        # Gradient step
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        if i>500:
            break

    plot_reconstruction(multi_autoencoder, (img_modalities, label),
                        indices, batch_size, epoch, batch_limit=3)

    print("Epoch {0}, {1} : {2}".format(epoch, timeSince(start), loss.data.view(-1).numpy()[0]))




#%% Test the plot
for i, sample in enumerate(train_loader):
    print(modalities)
    if i>1:
        break


indices = [2]
plot_reconstruction(multi_autoencoder, sample, indices, batch_size, batch_limit=3, plot=True)


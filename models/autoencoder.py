import torch.nn as nn
import torch.nn.functional as F
import torch

class Autoencoder(nn.Module):

    def __init__(self, input_size, layer_1_size, layer_2_size, layer_3_size, output_size):
        super(Autoencoder, self).__init__()

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_size, layer_1_size)
        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.fc3 = nn.Linear(layer_2_size, layer_3_size)
        self.fc4 = nn.Linear(layer_3_size, output_size)

    def encoder(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

    def decoder(self, x):
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




## Multi Autoencoder
class Encoder(nn.Module):

    def __init__(self, input_size, layer_1_size, layer_2_size, latent_size):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_size, layer_1_size)
        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.fc3 = nn.Linear(layer_2_size, latent_size)

    def encoder(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class Decoder(nn.Module):

    def __init__(self, latent_size, layer_1_size, layer_2_size, output_size):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(latent_size, layer_1_size)
        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.fc3 = nn.Linear(layer_1_size, output_size)

    def decoder(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class MultiAutoencoder(nn.Module):
    def __init__(self, input_size, encoder_layer_1_size, encoder_layer_2_size,
                 latent_layer_size,
                 decoder_layer_1_size, decoder_layer_2_size,
                 N):
        super(MultiAutoencoder, self).__init__()

        self.fc1 = nn.Linear(1, 1)

        self.N = N
        self.encoders = [Encoder(input_size,
                                 encoder_layer_1_size, encoder_layer_2_size,
                                 latent_layer_size) for i in range(N)]
        self.decoders = [Decoder(latent_layer_size,
                                 decoder_layer_1_size, decoder_layer_2_size,
                                 input_size) for i in range(N)]

    def encoder(self, multi_x):
        x = [self.encoders[i].encoder(multi_x[i]) for i in range(self.N)]
        return x

    @staticmethod
    def fusion(multi_x):
        x = torch.cat(multi_x)
        #x = x.max(dim=0)[0]
        x = x.mean(dim=0)
        return x

    def latent(self, multi_x):
        x = self.encoder(multi_x)
        x = self.fusion(x)
        return x

    def decoder(self, latent_x):
        multi_x = [self.decoders[i].decoder(latent_x) for i in range(self.N)]
        return multi_x

    def forward(self, x):
        x = self.encoder(x)
        x = self.fusion(x)
        x = self.decoder(x)
        return x
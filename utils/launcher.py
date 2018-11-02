import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt

import numpy as np

from utils.time import timeSince


def launcher_predict(net, data_generator, criterion, optimizer, n_epoch, n_iter, idx, plot=False):


    if plot:
        losses = []
        fig, ax = plt.subplots(1,1)


    idx_in, idx_target = idx
    start = time.time()

    for epoch in range(n_epoch):
        for i in range(n_iter):

            z = np.random.normal(size=(data_generator.dim_z, 1))
            res = data_generator.sample(z)

            # Training iteration
            value = res[idx_in]
            target = res[idx_target]
            #target = torch.FloatTensor(z).view(1 ,1, -1)

            # Initialize hidden, grad, loss
            net.zero_grad()
            loss = 0

            input = value.view(1, 1, -1)
            target = target.view(1, 1, -1)

            output = net(input)
            loss = criterion(output, target)

            # Gradient step
            loss.backward()
            optimizer.step()

        if plot:
            losses.append(loss.data.numpy())
            ax.plot(losses)
            plt.draw()
            plt.show()
            plt.pause(0.01)

        print("Epoch {0}, {1} : {2}".format(epoch, timeSince(start), loss.data.view(-1).numpy()[0]))

import numpy as np
from scipy.stats import ortho_group
import torch


class MatrixProjection:
    """
    Creates samples of the form X=AZ, Y=BZ, projections of Z
    Z is of dimension dim and projections are of dimension dim_input, for now the same for everyone.
    """

    def __init__(self, dim_z, dim_input, N, noise_variance=0.1):
        self.projection_matrix = [np.random.normal(size=(dim_input, dim_z)) for i in range(N)]
        self.noise_variance = noise_variance
        self.dim_input = dim_input
        self.dim_z = dim_z
        self.N = N

    def sample(self, z, noise = True):
        return [torch.FloatTensor(x.dot(z)+(1.*noise*np.random.normal(scale=self.noise_variance,
                                                    size=(self.dim_input, 1)))).view(1, 1, -1) for x in self.projection_matrix]


class RandomMatrixProjection:
    """
    Creates samples of the form X=AZ, Y=BZ, projections of Z
    Z is of dimension dim and projections are of dimension dim_input, for now the same for everyone.
    Projections are sub-projections of one orthogonal projection of O(dim_z)
    """

    def __init__(self, dim_z, dim_input, N, noise_variance = 0.1):
        self.projection_matrix = ortho_group.rvs(dim_z)
        self.noise_variance = noise_variance
        self.dim_input = dim_input
        self.dim_z = dim_z
        self.N = N

    def sample(self, z, noise=True):
        # Orthogonal Projection
        res = self.projection_matrix.dot(z)
        # Separate dimensions of the projection
        grid = np.floor(np.linspace(0, self.dim_z, self.N+2)).astype(int)
        modalities = [res[grid[i]:grid[i]+self.dim_input, :] for i in range(self.N)]
        if noise:
            modalities = [modality+np.random.normal(scale=self.noise_variance,
                                                    size=modality.shape) for modality in modalities]

        modalities = [torch.FloatTensor(modality).view(1, 1, -1) for modality in modalities]
        return modalities





class Toy3_nonlinear:
    """
    From a z = (z_1, z_2, z_3, z_4) generate x1, x2, x3 subsets with each of the possible permutations
    """

    def __init__(self, dim_input = 1, noise_variance=0.1):
        self.noise_variance = noise_variance
        self.dim_input = dim_input

    def sample(self, z=None, noise=True, batch_size=1):

        if z is None:
            x = np.random.normal(size=(batch_size, self.dim_input))
            y = np.random.normal(size=(batch_size, self.dim_input))
        else:
            x, y = z

        sample = [torch.FloatTensor(x*x*x +
                                  1.* noise * np.random.normal(scale=self.noise_variance,
                                                               size=(batch_size, 1))).view(batch_size, 1, -1),
                torch.FloatTensor(y*y*y +
                                  1. * noise * np.random.normal(scale=self.noise_variance,
                                                                size=(batch_size, 1))).view(batch_size, 1, -1),
                torch.FloatTensor(x+y +
                                  1. * noise * np.random.normal(scale=self.noise_variance,
                                                                size=(batch_size, 1))).view(batch_size, 1, -1)]
        return (x, y), sample


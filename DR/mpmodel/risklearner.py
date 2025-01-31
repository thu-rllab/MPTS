import torch
import torch.nn as nn
import pdb
from torch.distributions import Normal
from mpmodel.backbone_risklearner import Encoder, MuSigmaEncoder, Decoder

# Define RiskLearner for the current task distribution and the current backbone parameters...
class RiskLearner(nn.Module):
    """
    Implements risklearner for functions of arbitrary dimensions.
    x_dim : int Dimension of x values.
    y_dim : int Dimension of y values.
    r_dim : int Dimension of output representation r.
    z_dim : int Dimension of latent variable z.
    h_dim : int Dimension of hidden layer in encoder and decoder.
    """

    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim):
        super(RiskLearner, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Initialize networks
        self.xy_to_r = Encoder(x_dim, y_dim, h_dim, r_dim)
        self.r_to_mu_sigma = MuSigmaEncoder(r_dim, z_dim)
        self.xz_to_y = Decoder(x_dim, z_dim, h_dim, y_dim)

    def aggregate(self, r_i):
        return torch.mean(r_i, dim=1)

    def xy_to_mu_sigma(self, x, y):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.
        """
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)
        batch_size, num_points, _ = x.size()
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)

        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        return self.r_to_mu_sigma(r)

    def forward(self, x, y, output_type):
        """
        returns a distribution over target points y_target. We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)
        batch_size, num, x_dim = x.size()
        _, _, y_dim = y.size()

        if self.training:
            mu, sigma = self.xy_to_mu_sigma(x, y)
            z_variational_posterior = Normal(mu, sigma)
            z_sample = z_variational_posterior.rsample([1])

            p_y_pred = self.xz_to_y(x, z_sample, output_type)
            return p_y_pred, z_variational_posterior


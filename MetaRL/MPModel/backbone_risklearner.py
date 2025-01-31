import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.distributions import Normal


# Basic functions in RiskLearner
class Encoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i."""

    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)


class MuSigmaEncoder(nn.Module):
    """Maps r to z."""

    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class Decoder(nn.Module):
    """Maps (x+z) to y."""

    def __init__(self, x_dim, z_dim, h_dim, y_dim):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim

        layers = [nn.Linear(x_dim + z_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, y_dim)

    def forward(self, x, z, output_type):
        # batch_size=1

        batch_size, num_points, _ = x.size()
        num_repeat, _, _ = z.size()

        if batch_size == 1:
            x = x.repeat(num_repeat, 1, 1)
            z = z.repeat(1, num_points, 1)

            x_flat = x.view(num_repeat * num_points, self.x_dim)
            z_flat = z.view(num_repeat * num_points, self.z_dim)

            input_pairs = torch.cat((x_flat, z_flat), dim=1)
            hidden = self.xz_to_hidden(input_pairs)
            mu = self.hidden_to_mu(hidden)
            pre_sigma = self.hidden_to_sigma(hidden)

            mu = mu.view(num_repeat, num_points, self.y_dim)

            if output_type == "probabilistic":
                pre_sigma = pre_sigma.view(num_repeat, num_points, self.y_dim)
                sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
                p_y_pred = Normal(mu, sigma)
                p_y_pred = p_y_pred.rsample([1])[0]
            else:
                p_y_pred = mu

            return p_y_pred


class Ensemble_Predictor(nn.Module):
    def __init__(self, x_dim, h_dim, y_dim):
        super(Ensemble_Predictor, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim

        layers = [nn.Linear(x_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, y_dim)

    def forward(self, x,output_type):
        # batch_size=1

        batch_size, num_points, _ = x.size()

        if batch_size == 1:

            x_flat = x.view(num_points, self.x_dim)

            input_pairs = x_flat
            hidden = self.xz_to_hidden(input_pairs)
            mu = self.hidden_to_mu(hidden)
            pre_sigma = self.hidden_to_sigma(hidden)

            mu = mu.view(1, num_points, self.y_dim)

            if output_type == "probabilistic":
                pre_sigma = pre_sigma.view(1, num_points, self.y_dim)
                sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
                p_y_pred = Normal(mu, sigma)
                p_y_pred = p_y_pred.rsample([1])[0]
            else:
                p_y_pred = mu

            return p_y_pred
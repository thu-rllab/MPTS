import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import numpy as np


class RiskLearnerTrainer():
    """
    Class to handle training of RiskLearner for functions.
    """

    def __init__(self, device, risklearner, optimizer):

        self.device = device
        self.risklearner = risklearner
        self.optimizer = optimizer

        # ++++++Prediction distribution p(l|tau)++++++++++++++++++++++++++++
        self.output_type = "deterministic"

        # ++++++initialize the p(z_0)++++++++++++++++++++++++++++
        r_dim = self.risklearner.r_dim
        prior_init_mu = torch.zeros([1, r_dim]).to(self.device)
        prior_init_sigma = torch.ones([1, r_dim]).to(self.device)
        self.z_prior = Normal(prior_init_mu, prior_init_sigma)
        self.last_risk_x = None
        self.last_risk_y = None

        # ++++++Acquisition functions++++++++++++++++++++++++++++
        self.num_samples = 20

    def train(self, Risk_X, Risk_Y):
        Risk_X, Risk_Y = Risk_X.unsqueeze(0), Risk_Y.unsqueeze(0).unsqueeze(-1)
        # shape: batch_size, num_points, dim

        self.optimizer.zero_grad()
        p_y_pred, z_variational_posterior = self.risklearner(Risk_X, Risk_Y, self.output_type)
        z_prior = self.z_prior

        loss = self._loss(p_y_pred, Risk_Y, z_variational_posterior, z_prior)
        loss.backward()
        self.optimizer.step()

        # updated z_prior
        self.z_prior = Normal(z_variational_posterior.loc.detach(), z_variational_posterior.scale.detach())
        self.last_risk_x = Risk_X
        self.last_risk_y = Risk_Y

        return loss

    def negative_log_likelihood(self, y_true, mu, sigma):
        # Avoid division by zero
        sigma = torch.clamp(sigma, min=1e-6)
        # Compute the negative log likelihood
        loss = 0.5 * torch.log(sigma ** 2) + 0.5 * ((y_true - mu) ** 2) / (sigma ** 2)
        return loss.mean()  # taking mean over all samples if batched

    def _loss(self, p_y_pred, y_target, posterior, prior):

        negative_log_likelihood = F.mse_loss(p_y_pred.mean(0, keepdim=True), y_target, reduction="sum")

        # KL has shape (batch_size, r_dim). Take mean over batch and sum over r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(posterior, prior).mean(dim=0).sum()

        return negative_log_likelihood + kl

    def acquisition_function(self, Risk_X_candidate, gamma_mu, gamma_sigma):

        Risk_X_candidate = Risk_X_candidate.to(self.device)
        x = Risk_X_candidate.unsqueeze(0)
        # Shape: 1 * 100 * 2
        
        if self.last_risk_x is None:
            z_sample = self.z_prior.rsample([self.num_samples])
        else:
            _, z_variational_posterior = self.risklearner(self.last_risk_x, self.last_risk_y, self.output_type)
            z_posterior = Normal(z_variational_posterior.loc.detach(), z_variational_posterior.scale.detach())
            z_sample = z_posterior.rsample([self.num_samples])   # sampling from prior
        # Shape: num_samples * 1 * 10

        p_y_pred = self.risklearner.xz_to_y(x, z_sample, self.output_type)
        # Shape: num_samples * num_candidate * 1

        output_mu = torch.mean(p_y_pred, dim=0)
        output_sigma = torch.std(p_y_pred, dim=0)
        acquisition_score = gamma_mu * output_mu + gamma_sigma * output_sigma
        return acquisition_score, p_y_pred

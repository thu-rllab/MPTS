import torch
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

from torch import nn
import numpy as np
import torch.nn.functional as F

import copy 

from rl_utils.optimization import conjugate_gradient
from rl_utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize)

import scipy.stats as st


##########################################################################################################################
    # This part is to introduce the MLP for the implementation of MAML. --> Meta Player
##########################################################################################################################
def get_task_returns(episodes_per_task, use_last_r=False):
    # sum up for each rollout, then take the mean across rollouts
    returns = []
    # for training, episodes_per_task [(training episodes, validation episodes)], shape [num_tasks]
    # for testing, episodes_per_task [num_test_steps+1 episodes], shape [num_tasks]
    for task_idx in range(len(episodes_per_task)):
        curr_returns = []
        # for training, episodes (training episodes, validation episodes), tuple shape [2]
        # for testing, episodes [num_test_steps+1 episodes], list length: num_test_steps+1
        episodes = episodes_per_task[task_idx]
        for update_idx in range(len(episodes)):
            # compute returns for individual rollouts
            # rewards: [num_steps, num_eval_traj,], mask: [num_steps, num_eval_traj,]
            # ret = (episodes[update_idx].rewards * episodes[update_idx].mask).sum(dim=0)

            #last reward
            if use_last_r:
                ret = []
                for i in range(len(episodes[update_idx].infos)):
                    ret.append(episodes[update_idx].infos[i][-1]['reward_forward'] if 'reward_forward' in episodes[update_idx].infos[i][-1].keys() else episodes[update_idx].infos[i][-1]['reward_goal'])
                ret = torch.tensor(ret)
            else:
                ret = (episodes[update_idx].rewards * episodes[update_idx].mask).sum(dim=0)

            curr_returns.append(ret)
        # for training, curr_returns [training episodes rewards, fast adaptation episodes rewards], list shape [2], training episodes rewards with shape [num_eval_traj]
        # for testing, curr_returns [initial rewards,  fast adaptation episodes rewards of num_test_step updates], list shape [1+num_test_steps], episodes rewards with shape [num_eval_traj]

        # for training, result of torch.stack(curr_returns, dim=1) will be: [num_eval_traj, 2], here 2 means the training/validation
        # for testing, result of torch.stack(curr_returns, dim=1) will be: [num_eval_traj, num_test_steps+1]
        returns.append(torch.stack(curr_returns, dim=1))

    returns = torch.stack(returns) # for training, returns shape [num_tasks, num_eval_traj, 2]; for testing, returns shape [num_tasks, num_eval_traj, num_test_steps+1]
    # returns = returns.reshape((-1, returns.shape[-1])) # for training, returns shape [num_tasks*num_eval_traj, 2]; for testing, returns shape [num_tasks*num_eval_traj, num_test_steps+1]
    returns = returns.mean(dim=1) # for training, returns shape [num_tasks, 2]; for testing, returns shape [num_tasks, num_test_steps+1]
    return returns


class MetaLearner(object):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes 
    (before and after the one-step adaptation), compute the inner loss, compute 
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, 
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized 
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, 
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    """

    def __init__(self, args, sampler, policy, baseline, gamma=0.95,
                 fast_lr=0.5, tau=1.0, device='cpu'):
        self.args = args
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.to(device)

    def inner_loss(self, episodes, params=None):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)

        loss = -weighted_mean(log_probs * advantages, dim=0, weights=episodes.mask)

        return loss

    def adapt(self, episodes, first_order=False, params=None, lr=None, update=True):
        """Adapt the parameters of the policy network to a new task, from 
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """

        if lr is None:
            lr = self.fast_lr

        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)

        # Get the loss on the training episodes
        loss = self.inner_loss(episodes, params=params)

        # Get the new parameters after a one-step gradient update
        if update:
            params = self.policy.update_params(loss, step_size=lr, first_order=first_order, params=params)

            return params, loss
        else:
            return loss

    def sample(self, tasks, first_order=False):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        episodes = []
        losses = []
        after_losses = []
        for task in tasks:
            self.sampler.reset_task(task)
            self.policy.reset_context()
            train_episodes = self.sampler.sample(self.policy, gamma=self.gamma)
            # inner loop (for CAVIA, this only updates the context parameters)
            params, loss = self.adapt(train_episodes, first_order=first_order)
            # rollouts after inner loop update
            valid_episodes = self.sampler.sample(self.policy, params=params, gamma=self.gamma)
            # support/training episodes and query/testing (after adaptation) episodes
            episodes.append((train_episodes, valid_episodes))
            losses.append(loss.item())
            after_losses.append(self.adapt(valid_episodes, params=params, update=False).item())
        
        # episodes shape [num_tasks], losses shape [num_tasks], here num_tasks means the meta task batch size
        # element in episodes is a tuple (meta training episode, fast adaptation episode)
        return episodes, losses, after_losses

    def test(self, tasks, num_steps, batch_size, halve_lr):
        """Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.batchsize
        """
        episodes_per_task = []
        for task in tasks:

            # reset context params (for cavia) and task
            self.policy.reset_context()
            self.sampler.reset_task(task)

            # start with blank params
            params = None

            # gather some initial experience and log performance
            test_episodes = self.sampler.sample(self.policy, gamma=self.gamma, params=params, batch_size=batch_size, test=True)

            # initialise list which will log all rollouts for the current task
            curr_episodes = [test_episodes]

            for i in range(1, num_steps + 1):

                # lower learning rate after first update (for MAML, as described in their paper)
                if i == 1 and halve_lr:
                    lr = self.fast_lr / 2
                else:
                    lr = self.fast_lr

                # inner-loop update
                params, loss = self.adapt(test_episodes, first_order=True, params=params, lr=lr)

                # get new rollouts
                test_episodes = self.sampler.sample(self.policy, gamma=self.gamma, params=params, batch_size=batch_size, test=True)
                curr_episodes.append(test_episodes)

            episodes_per_task.append(curr_episodes)

        self.policy.reset_context()
        return episodes_per_task

    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):

            # this is the inner-loop update
            self.policy.reset_context()
            params, _ = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""

        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    def surrogate_loss(self, episodes, old_pis=None, mean_loss=True):
        # mean_loss: mean over the num_tasks
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):

            # do inner-loop update
            self.policy.reset_context()
            params, _ = self.adapt(train_episodes)

            with torch.set_grad_enabled(old_pi is None):

                # get action values after inner-loop update
                pi = self.policy(valid_episodes.observations, params=params)
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                values = self.baseline(valid_episodes)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages, weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions)
                             - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(ratio * advantages, dim=0, weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
                kls.append(kl)
        
        if mean_loss:
            # mean operation over the num_tasks dim
            return torch.mean(torch.stack(losses, dim=0)), torch.mean(torch.stack(kls, dim=0)), pis
        else:
            # retain both the mean loss and num_tasks shape loss
            return torch.mean(torch.stack(losses, dim=0)), torch.mean(torch.stack(kls, dim=0)), \
                   torch.stack(losses, dim=0), torch.stack(kls, dim=0), pis

    def step(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5, mean_loss=True, cvar_conf=0.0, game_framework=0):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        mean_loss: if false, then return both mean and task batch shape loss
        cvar_conf: screen int((1.0-cvar_conf)*task batch)) worst for meta updates
        """
        if mean_loss == True:
            # vanilla maml cases
            old_loss, _, old_pis = self.surrogate_loss(episodes, mean_loss=mean_loss)
        elif cvar_conf == 0.0 and game_framework == 0:
            # game theoretical (adversarially robust) maml cases
            old_loss, _, batch_old_loss, batch_, old_pis = self.surrogate_loss(episodes, mean_loss=mean_loss)
        elif cvar_conf == 0.0 and game_framework == 3:
            # group dro maml cases
            old_loss, _, batch_old_loss, batch_, old_pis = self.surrogate_loss(episodes, mean_loss=mean_loss)
            curr_returns = get_task_returns(episodes, use_last_r=self.args.baseline_use_last_r)
                    
            adv_probs = torch.ones(1).cuda()
            adv_probs = adv_probs * torch.exp(self.args.groupdro_weight * (-curr_returns[:,-1].to('cuda').float()))
            adv_probs = adv_probs/torch.sum(adv_probs)
            old_loss = torch.dot(batch_old_loss, adv_probs)
        else:
            # distributionally robust maml with cvar principle cases
            old_loss, _, batch_old_loss, batch_, old_pis = self.surrogate_loss(episodes, mean_loss=mean_loss)
            curr_returns = get_task_returns(episodes, use_last_r=self.args.baseline_use_last_r)
            # outer_eval_loss_arr = batch_old_loss.cpu().detach().numpy()#bs,
            outer_eval_loss_arr = -curr_returns[:,-1].cpu().detach().numpy()#bs,
            
            # argmax indices and screen the proportional worst task episodes
            topk_indices = (-outer_eval_loss_arr).argsort()[:int((1.0-cvar_conf)*batch_old_loss.size()[0])]
            topk_indices_list = topk_indices.tolist()
            topk_episodes = [episodes[i] for i in topk_indices_list]
            
            # cvar batch old loss computation, average over screened tasks, cvar top batch old policies
            old_loss = torch.mean(batch_old_loss[topk_indices_list])
            old_pis = [old_pis[i] for i in topk_indices_list]
            
        # this part will take higher order gradients through the inner loop:
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        if mean_loss == True:
            # vanilla maml cases
            hessian_vector_product = self.hessian_vector_product(episodes, damping=cg_damping)
        elif cvar_conf == 0.0:
            # game theoretical (adversarially robust) maml cases/group dro maml cases
            hessian_vector_product = self.hessian_vector_product(episodes, damping=cg_damping)
        else:
            # distributionally robust maml with cvar principle cases
            hessian_vector_product = self.hessian_vector_product(topk_episodes, damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step, self.policy.parameters())
            if mean_loss == True:
                # vanilla maml cases 
                # fast adaptation with support and eval with query
                loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis, mean_loss=mean_loss)
            elif cvar_conf == 0.0:
                # game theoretical (adversarially robust) maml cases/group dro maml cases
                # fast adaptation with support and eval with query
                loss, kl, batch_loss, batch_kl, _ = self.surrogate_loss(episodes, old_pis=old_pis, mean_loss=mean_loss)
            else:
                # distributionally robust maml with cvar principle cases            
                loss, kl, batch_loss, batch_kl, _ = self.surrogate_loss(topk_episodes, old_pis=old_pis, mean_loss=mean_loss)
                
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            print('no update?')
            vector_to_parameters(old_params, self.policy.parameters())
        
        if mean_loss == True:
            return loss
        else:
            # mean over task batch adaptation evaluation loss, task batch adaptation evaluation loss
            return loss, batch_loss

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.device = device
      
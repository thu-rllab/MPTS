import multiprocessing as mp

import gym
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MP_BatchSampler(object):
    def __init__(self, args,risk_learner_trainer, gamma_0, gamma_1):
        self.risklearner_trainer = risk_learner_trainer
        self.args = args
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        self.current_epoch = 0

    def get_acquisition_score(self, tasks):
        acquisition_score, acquisition_mean, acquisition_std = self.risklearner_trainer.acquisition_function(tasks, self.gamma_0, self.gamma_1)
        return acquisition_score, acquisition_mean, acquisition_std

    def sample_tasks(self, shape, multiplier, init_dist='Uniform', test=False):
        candidate_tasks = torch.rand(int(multiplier*shape[0]*shape[1]),shape[2])
        # candidate_tasks = np.random.uniform(0.0, 1.0, size=(int(multiplier*shape[0]*shape[1]),shape[2]))
        acquisition_score, acquisition_mean, acquisition_std = self.get_acquisition_score(candidate_tasks) # candidate tasks 15 * loss 1
        acquisition_score = acquisition_score.squeeze(1) # candidate tasks 15
        if not self.args.no_add_random:
            selected_values, selected_index = torch.topk(acquisition_score, k=shape[0]*shape[1]//2)
        else:
            selected_values, selected_index = torch.topk(acquisition_score, k=shape[0]*shape[1])
        mask = ~torch.isin(torch.arange(0, int(multiplier*shape[0]*shape[1])), selected_index.cpu())
        unselected_index = torch.arange(0, int(multiplier*shape[0]*shape[1]))[mask]
        index=torch.cat((selected_index.cpu(),unselected_index),dim=0)[:shape[0]*shape[1]][torch.randperm(shape[0]*shape[1])] # num_tasks 10
        index = index.cpu()
        tasks = candidate_tasks[index]
        tasks = tasks.view(shape[0],shape[1],shape[2]).numpy()

        return tasks
    
    def train(self, tasks, y):
        loss, recon_loss, kl_loss = self.risklearner_trainer.train(tasks, y)
        return loss, recon_loss, kl_loss
    
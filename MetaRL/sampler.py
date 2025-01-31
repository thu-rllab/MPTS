import multiprocessing as mp
import numpy as np
import gym
import torch
import wandb
from envs.subproc_vec_env import SubprocVecEnv
from episode import BatchEpisodes

def make_env(env_name):
    def _make_env():
        return gym.make(env_name)

    return _make_env


class BatchSampler(object):
    def __init__(self, env_name, batch_size, device, seed, num_workers=mp.cpu_count() - 1):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(self.num_workers)], queue=self.queue)
        self.envs.seed(seed)
        self._env = gym.make(env_name)
        self._env.seed(seed)
        self.total_t = 0
    def close(self):
        self.envs.close()
        
    def sample(self, policy, params=None, gamma=0.95, batch_size=None, test=False):
        if batch_size is None:
            batch_size = self.batch_size
        episodes = BatchEpisodes(batch_size=batch_size, gamma=gamma, device=self.device)
        for i in range(batch_size):
            self.queue.put(i)
        for _ in range(batch_size):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset(batch_size)
        dones = [False]
        interact_steps = 0
        interact_steps += batch_size
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=self.device)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, infos = self.envs.step(actions, batch_size)
            sort_batch_ids = [i if batch_ids[i] is not None else None for i in range(batch_size)]
            episodes.append(observations, actions, rewards, sort_batch_ids, infos, dones)
            observations, batch_ids = new_observations, new_batch_ids
            interact_steps += batch_size
            interact_steps -= np.sum(dones)
        if not test:
            self.total_t += interact_steps
        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)
    
    def generate_tasks(self, norm_z):
        tasks = self._env.unwrapped.generate_tasks(norm_z)
        return tasks
        
    def sample_tasks(self, i_iter, num_tasks, init_dist, test=False):
        if test:
            candidate_tasks = self._env.unwrapped.grid_sample_tasks(num_tasks)
            return candidate_tasks
        tasks = self._env.unwrapped.sample_tasks(num_tasks, init_dist)
        return tasks

class MP_BatchSampler(BatchSampler):
    def __init__(self, args,risk_learner_trainer, gamma_0, gamma_1, env_name, batch_size, device, seed, num_workers=mp.cpu_count() - 1):
        # self.risk_learner = risk_learner.to(device)
        self.risklearner_trainer = risk_learner_trainer
        self.args = args
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        self.warmup = args.warmup
        self.current_epoch = 0
        
        super(MP_BatchSampler, self).__init__(env_name, batch_size, device, seed, num_workers)

    def identifier_preprocess(self, tasks):
        identifier_list = []
        for task in tasks:
            local_identifier_list = []
            for key in task.keys():
                local_identifier_list.append(torch.tensor(task[key]).float())
            if len(local_identifier_list) > 1:
                local_identifier = torch.stack(local_identifier_list, dim=0)
            else:
                local_identifier = local_identifier_list[0]
            identifier_list.append(local_identifier)
        candidate_identifier = torch.stack(identifier_list, dim=0)
        return candidate_identifier
 
    def get_acquisition_score(self, i_iter, tasks):
        candidate_identifier = self.identifier_preprocess(tasks)
        acquisition_score, acquisition_mean, acquisition_std = self.risklearner_trainer.acquisition_function(candidate_identifier, self.gamma_0, self.gamma_1)
        return acquisition_score, acquisition_mean, acquisition_std

    def sample_tasks(self, i_iter, num_tasks, init_dist, test=False):
        if test:
            # candidate_tasks = self._env.unwrapped.sample_tasks(num_tasks, init_dist)
            candidate_tasks = self._env.unwrapped.grid_sample_tasks(num_tasks)
            return candidate_tasks
        if self.current_epoch < self.warmup:
            candidate_tasks = self._env.unwrapped.sample_tasks(num_tasks, init_dist)
            self.current_epoch += 1
            return candidate_tasks
        candidate_tasks = self._env.unwrapped.sample_tasks(int(num_tasks*self.args.sample_ratio), init_dist)
        acquisition_score, acquisition_mean, acquisition_std = self.get_acquisition_score(i_iter, candidate_tasks)
        acquisition_score = acquisition_score.squeeze(1)
        if self.args.add_random:
            _, selected_index = torch.topk(acquisition_score, k=int(num_tasks*(1-self.args.random_ratio)))
        else:
            _, selected_index = torch.topk(acquisition_score, k=num_tasks)
        selected_index = selected_index.cpu()
        tasks = [candidate_tasks[i] for i in selected_index]
        if self.args.add_random:
            for i in range(len(candidate_tasks)):
                if len(tasks) >= num_tasks:
                    break
                tasks.append(candidate_tasks[i])
        return tasks
    
    def train(self, tasks, y):
        x = self.identifier_preprocess(tasks).to(y.device)#n_tasks
        if len(y.shape) == 2:
            y = torch.mean(y, dim=-1).reshape(-1).float()
        loss, recon_loss, kl_loss = self.risklearner_trainer.train(x, y)
        return loss, recon_loss, kl_loss
    
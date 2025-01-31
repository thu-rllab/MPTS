import datetime
import json
import os
# import matplotlib.pyplot as plt
import time

import numpy as np
import scipy.stats as st
import torch

import utils
from arguments import parse_args

from baseline import LinearFeatureBaseline
from metalearner import MetaLearner
from policies.categorical_mlp import CategoricalMLPPolicy
from policies.normal_mlp import NormalMLPPolicy
from sampler import BatchSampler
from metalearner import MetaLearner


def get_returns(episodes_per_task):

    # sum up for each rollout, then take the mean across rollouts
    returns = []
    for task_idx in range(len(episodes_per_task)):
        curr_returns = []
        episodes = episodes_per_task[task_idx]
        for update_idx in range(len(episodes)):
            # compute returns for individual rollouts
            ret = (episodes[update_idx].rewards * episodes[update_idx].mask).sum(dim=0)
            curr_returns.append(ret)
        # result will be: num_evals * num_updates
        returns.append(torch.stack(curr_returns, dim=1))

    # result will be: num_tasks * num_evals * num_updates
    returns = torch.stack(returns)
    returns = returns.reshape((-1, returns.shape[-1]))

    return returns


def total_rewards(episodes_per_task, interval=False):

    returns = get_returns(episodes_per_task).cpu().numpy()

    mean = np.mean(returns, axis=0) # output shape (2, )
    conf_int = st.t.interval(0.95, len(mean) - 1, loc=mean, scale=st.sem(returns, axis=0))
    conf_int = mean - conf_int
    if interval:
        return mean, conf_int[0]
    else:
        return mean




######################################eval_returns_per_task######################################


def task_get_returns(episodes_per_task):

    # sum up for each rollout, then take the mean across rollouts
    returns = []
    for task_idx in range(len(episodes_per_task)):
        curr_returns = []
        episodes = episodes_per_task[task_idx]
        for update_idx in range(len(episodes)):
            # compute returns for individual rollouts
            ret = (episodes[update_idx].rewards * episodes[update_idx].mask).sum(dim=0)
            curr_returns.append(ret)
        # result will be: num_evals * num_updates
        returns.append(torch.stack(curr_returns, dim=1))

    # result will be: num_tasks * num_evals * num_updates
    returns = torch.stack(returns)
    adapt_returns = returns[...,-1]
    adapt_returns = adapt_returns.mean(dim=-1) # average over horizon, output shape [num_tasks]

    return adapt_returns


def task_total_rewards(episodes_per_task, conf_level, return_all=False):

    adapt_returns = task_get_returns(episodes_per_task).cpu().numpy()
    
    # print('adapt_returns', adapt_returns)
    topk_indices = adapt_returns.argsort()[:int((1.0-args.conf_level)*adapt_returns.shape[0])]
    topk_returns = adapt_returns[topk_indices]
    
    avg_returns = np.mean(adapt_returns, axis=0)
    cvar_returns = np.mean(topk_returns, axis=0)
    worst_returns = np.min(adapt_returns, axis=0)


    alpha = [0.95, 0.9, 0.7, 0.5, 0.3, 0.1]
    cvar_adapt_returns = []
    for i in alpha:
        cvar_ind = adapt_returns.argsort()[:int((1-i)*adapt_returns.shape[0])]
        cvar_ind_list = cvar_ind.tolist()
        cvar_adapt_returns.append(np.mean(adapt_returns[cvar_ind_list]))
    
    if return_all:
        return avg_returns, cvar_returns, worst_returns, adapt_returns
    else:
        return avg_returns, cvar_returns, worst_returns, cvar_adapt_returns
    



#################################################################################################




def main(args, seeds):
    print('starting....')
    print('transformed_dis', args.transformed_dis)


    
    continuous_actions = (args.env_name in ['HalfCheetahVel-v0', 
                                            'HalfCheetahMassVel-v1',    
                                            'Walker2dMassVel-v1',
                                            'Walker2dVel-v1',
                                            'ReacherPos-v0',
                                            ])  


    sampler = BatchSampler(args.env_name, 
                           batch_size=args.fast_batch_size, 
                           num_workers=args.num_workers,
                           device=args.device, seed=args.seed)

    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers
            )
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)

    # initialise baseline
    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))
    avg_returns_list = []
    seed_list = []
    testing_epoch_list = []
    cvar_returns_list = []
    worst_returns_list = []

    for seed in seeds:
        args.seed = seed
        utils.set_seed(args.seed, cudnn=args.make_deterministic)
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        modelpathlist = os.listdir(os.path.join(args.save_model_dir,args.env_name))
        for modelpath in modelpathlist:
            if framework == 1 and '1__DR' in modelpath and f'seed{seed}' in modelpath:
                policy.load_state_dict(torch.load(modelpath + f'/policy-{args.meta_testing_epoch}.pt'))
            elif framework == 2 and '2__Vanilla' in modelpath and f'seed{seed}' in modelpath:
                policy.load_state_dict(torch.load(modelpath + f'/policy-{args.meta_testing_epoch}.pt'))
            elif framework == 3 and '3__GroupDRO' in modelpath and f'seed{seed}' in modelpath:
                policy.load_state_dict(torch.load(modelpath + f'/policy-{args.meta_testing_epoch}.pt'))
            elif framework == 4 and '2__task' in modelpath and f'seed{seed}' in modelpath:
                policy.load_state_dict(torch.load(modelpath + f'/policy-{args.meta_testing_epoch}.pt'))
        # initialise meta-learner
        metalearner = MetaLearner(args,
                                sampler, 
                                policy, 
                                baseline, 
                                gamma=args.gamma, 
                                fast_lr=args.fast_lr, 
                                tau=args.tau,
                                device=args.device)

        tasks = sampler.sample_tasks(0, 100, args.init_dist)

        # do the inner-loop update for each task
        # this returns training (before update) and validation (after update) episodes
        # 1 step in fast adaptation
        episodes, inner_losses, after_inner_losses = metalearner.sample(tasks, first_order=args.first_order)

        avg_returns, cvar_returns, worst_returns, adapt_returns = task_total_rewards(episodes, conf_level=args.conf_level, return_all=False)

        print ('seed',args.seed,'Avg return after update: ', avg_returns)
        print ('seed',args.seed,'Worst return after update: ', worst_returns)
        print ('seed',args.seed,'CVaR list return after update: ', adapt_returns)
        avg_returns_list.append(avg_returns)
        seed_list.append(args.seed)
        cvar_returns_list.append(adapt_returns)
        worst_returns_list.append(worst_returns)
        # utils.set_seed(None)
    sampler.close()
    return seed_list, avg_returns_list, cvar_returns_list, worst_returns_list
    

if __name__ == '__main__':
    envs = ['HalfCheetahMassVel-v1', 'HalfCheetahVel-v0', 'Walker2dMassVel-v1', 'Walker2dVel-v1', 'ReacherPos-v0']
    meta_testing_epoches = [500, 500, 500, 500, 250]
    frameworks = [1,2,3,4]
    seeds = [10,20,30,40,50,60,70]
    import csv
    for idx, env in enumerate(envs):
        for framework in frameworks:
            for i in range(10):
                args = parse_args()
                args.env_name = env
                args.meta_testing_epoch = meta_testing_epoches[idx] - i
                args.game_framework = framework
                seed_list, avg_returns_list, cvar_returns_list, worst_returns_list = main(args, seeds)
                #save to csv
                with open('metatesting_results.csv', mode='a') as file:
                    writer = csv.writer(file)
                    writer.writerow([env, framework, meta_testing_epoches[idx] - i, seed_list, avg_returns_list, cvar_returns_list, worst_returns_list])
                    print('saved to csv')

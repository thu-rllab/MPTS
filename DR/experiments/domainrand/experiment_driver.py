import matplotlib
matplotlib.use('Agg')

import random
import logging

import numpy as np
import torch
import gym
import argparse
import os

from common.agents.ddpg.ddpg import DDPG
from common.agents.ddpg_actor import DDPGActor
from common.utils.visualization import Visualizer
from common.utils.sim_agent_helper import generate_simulator_agent
from common.utils.logging import setup_experiment_logs, reshow_hyperparameters, StatsLogger

from experiments.domainrand.args import get_args, check_args
# from torch.utils.tensorboard import SummaryWriter
import datetime
import json
import wandb
def snapshot_src(src, target, exclude_from):
    try:
        os.mkdir(target)
    except OSError:
        pass
    os.system(f"rsync -rv --exclude-from={exclude_from} {src} {target}")


if __name__ == '__main__':
    args = get_args()
    check_args(args)
    if args.algo == 'drm':
        args.nagents = int(args.nagents / (1-args.cvar))
    
    work_dir = 'runs/{}/{}/addr{}_bn{}_dist{}_unif{}_0r{}_1r{}_slr{}_sstep{}_smul{}_sfreq{}_kl{}_s{}'.format(
     args.folder,args.randomized_eval_env_id, int(not args.no_add_random), int(not args.no_batch_norm), int(args.use_dist),
     args.uniform_sample_steps, args.sampling_gamma_0, args.sampling_gamma_1, args.sampler_lr, args.sampler_train_times, args.sampler_multiplier, args.svpg_rollout_length, args.kl_weight, args.seed)
    wandb.init(project=args.wandb_project, name=work_dir, config=vars(args))
    snapshot_src('.', os.path.join(work_dir, 'src'), '.gitignore')
    paths = setup_experiment_logs(args, work_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    stats_logger = StatsLogger(args)
    visualizer = Visualizer(randomized_env_id=args.randomized_eval_env_id, seed=args.seed)

    reference_env = gym.make(args.reference_env_id)

    if args.freeze_agent:
        # only need the actor
        agent_policy = DDPGActor(
            state_dim=reference_env.observation_space.shape[0], 
            action_dim=reference_env.action_space.shape[0], 
            agent_name=args.agent_name,
            load_agent=args.load_agent
        )
    else:
        agent_policy = DDPG(
            state_dim=reference_env.observation_space.shape[0], 
            action_dim=reference_env.action_space.shape[0], 
            agent_name=args.agent_name,
        )

        if args.load_agent: # no
            agent_policy.load_model()

    
    simulator_agent = generate_simulator_agent(args)

    svpg_timesteps = 0

    while simulator_agent.agent_timesteps < args.max_agent_timesteps:
        if svpg_timesteps % args.plot_frequency == 0:
            if not args.freeze_svpg and args.uniform_sample_steps<1:
                visualizer.plot_sampling_frequency(simulator_agent, agent_policy, 
                    simulator_agent.agent_timesteps, log_path=paths['sampling_logs'], plot_path=paths['sampling_plots'])
                
        logging.info("SVPG TS: {}, Agent TS: {}".format(svpg_timesteps, simulator_agent.agent_timesteps))
        
        solved, info = simulator_agent.select_action(agent_policy)
        svpg_timesteps += 1

    agent_policy.save(filename='final-seed{}'.format(args.seed), directory=paths['paper'])
    visualizer.plot_sampling_frequency(simulator_agent, agent_policy, 
        simulator_agent.agent_timesteps, log_path=paths['sampling_logs'], plot_path=paths['sampling_plots'])
    
    generalization_metric = visualizer.generate_ground_truth(simulator_agent, agent_policy, svpg_timesteps, 
        log_path=paths['groundtruth_logs'])
    np.savez('{}/generalization-seed{}.npz'.format(paths['paper'], args.seed),
        generalization_metric=generalization_metric,
        svpg_timesteps=svpg_timesteps,
        learning_curve_timesteps=simulator_agent.agent_timesteps
    )
    if 'Lunar' in args.randomized_eval_env_id:
        test_return_grid, test_tau_grid = simulator_agent.evaluate_in_full_range(agent_policy)
        np.savez('{}/gridtest-seed{}.npz'.format(paths['paper'], args.seed),
            test_return_grid=test_return_grid,
            test_tau_grid=test_tau_grid
        )
    reshow_hyperparameters(args, paths)

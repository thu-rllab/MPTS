import argparse
import multiprocessing as mp
import os
import warnings

import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Fast Context Adaptation via Meta-Learning (CAVIA)')
    
    parser.add_argument('--task', type=str, default='ar-navigation', help='problem setting: vanilla-navigation or ar-navigation')
    parser.add_argument('--sampling_strategy', type=str, default='random')
    parser.add_argument('--sampling_gamma_0', type=float, default=1.0)
    parser.add_argument('--sampling_gamma_1', type=float, default=5.0)

    parser.add_argument('--sample_ratio', type=float, default=1.5)
    parser.add_argument('--random_ratio', type=float, default=0.5)
    parser.add_argument('--meta_testing_epoch', type=int, default=500)
    parser.add_argument('--add_random', default=False, action='store_true')

    # parser.add_argument('--use_inner_loss', default=False, action='store_true')
    parser.add_argument('--use_last_r', default=True, action='store_true')
    # parser.add_argument('--baseline_use_last_r', default=False, action='store_true')

    # parser.add_argument('--batch_norm', default=True, action='store_false')
    parser.add_argument('--sampler_lr', default=0.005, type=float)
    parser.add_argument('--sampler_train_times', default=10, type=int)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--kl_weight', type=float, default=0.0001)
    parser.add_argument('--groupdro_weight', type=float, default=0.001)
    parser.add_argument('--output_type', type=str, default="deterministic")
    parser.add_argument('--save_model_dir', type=str, default="save_models")
    parser.add_argument('--wandb_project', type=str, default="metarl")
    parser.add_argument('--no_wandb_or_modelsave', default=False, action='store_true')


    parser.add_argument('--entropy_weight', type=float, default=0.2, help='entropy weight to constrain the distributional shifts')
    
    # General
    parser.add_argument('--env-name', type=str,
                        default='2DNavigation-v1',
                        help='name of the environment')
    parser.add_argument('--conf-level', type=float, default=0.5,
                        help='confidence level')    
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
                        help='use the first-order approximation of MAML/CAVIA')
    parser.add_argument('--num-context-params', type=int, default=0,
                        help='number of context parameters')

    # Run MAML instead of CAVIA
    parser.add_argument('--maml', action='store_true', default=True,
                        help='turn on MAML')
    parser.add_argument('--game_framework', type=int, default=0,
                        help='turn on 1: DR_MAML, 2: vanilla MAML, 3: GDRM')    
    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of hidden layers')

    parser.add_argument('--init_dist', type=str, default='Uniform', help='Uniform or Normal')    
    
    # Testing
    parser.add_argument('--test-freq', type=int, default=10,
                        help='How often to test multiple updates')
    parser.add_argument('--num-test-steps', type=int, default=1,
                        help='Number of inner loops in the test set')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help='batch size (number of trajectories) for testing')
    parser.add_argument('--halve-test-lr', action='store_true', default=False,
                        help='half LR at test time after one update')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
                        help='number of rollouts for each individual task ()')
    parser.add_argument('--fast-lr', type=float, default=0.1,
                        help='learning rate for the 1-step gradient update of MAML/CAVIA')
    
    parser.add_argument('--transformed_dis', action='store_true', default=False)

    # Optimization
    parser.add_argument('--num-batches', type=int, default=501,
                        help='number of iterations in meta training, default: 500')
    parser.add_argument('--meta-batch-size', type=int, default=20,
                        help='number of tasks per batch')
    parser.add_argument('--eval-meta-batch-size', type=int, default=40,
                        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
                        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
                        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
                        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
                        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
                        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--num-workers', type=int, default=min(mp.cpu_count() - 1, 100),
                        help='number of workers for trajectories sampling')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--make_deterministic', action='store_true',
                        help='make everything deterministic (set cudnn seed; num_workers=1; '
                             'will slow things down but make them reproducible!)')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])

    if args.make_deterministic:
        args.num_workers = 1

    # use the GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.output_folder = 'maml' if args.maml else 'cavia'

    if args.maml and not args.halve_test_lr:
        warnings.warn('You are using MAML and not halving the LR at test time!')

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')

    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    return args


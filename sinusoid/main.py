import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from Dataset.task_distribution import SineDistribution
from backbone_maml import SineNet
from trainer_maml import SineMAML
from utils import log_string
import random
import wandb

'''
 MAML sine backbone
# https://github.com/GauravIyer/MAML-Pytorch/blob/master/Experiment%201/Experiment_1_Sine_Regression.ipynb
'''

if __name__ == "__main__":



    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--log_name', type=str, nargs='?', default='log_mse/test_', help="log name")
    parser.add_argument('--sampling_strategy', type=str, nargs='?', default='random', help="sampling_strategy")
    parser.add_argument('--evaluation_strategy', type=str, nargs='?', default='mse', help="evaluation_strategy")
    parser.add_argument('--cvar_alpha', type=float, nargs='?', default=0.5, help="cvar_alpha")
    parser.add_argument('--global_seed', type=int, nargs='?', default=10, help="global seed")
    parser.add_argument('--risklearner_lr', type=float, nargs='?', default=0.0005, help="risklearner_lr")

    parser.add_argument('--num_candidates', type=int, nargs='?', default=32, help="number of candidates")

    parser.add_argument('--gamma_mu', type=float, nargs='?', default=1, help="acquisition function weight for mean")
    parser.add_argument('--gamma_sigma', type=float, nargs='?', default=3, help="acquisition function weight for sigma")
    args = parser.parse_args()

config = {}

config["gpu_id"] = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_id"]
config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config["global_seed"] = args.global_seed

random.seed(config["global_seed"])
np.random.seed(config["global_seed"])

torch.manual_seed(config["global_seed"])
torch.cuda.manual_seed(config["global_seed"])
torch.cuda.manual_seed_all(config["global_seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# maml hyparpameters
config["k"] = 10  # number of support points
config["alpha"] = 0.001  # inner loop
config["beta"] = 0.001 # outer loop
config["test_alpha"] = 0.001  # !!! SGD updating rather than selfupdating smaller lr more stable.
config["test_inner_steps"] = 10


# Sampling
config["num_metatasks"] = 16
config["num_candidates"] = args.num_candidates
config["risklearner_lr"] = args.risklearner_lr

config["gamma_mu"] = args.gamma_mu
config["gamma_sigma"] = args.gamma_sigma

config["sampling_strategy"] = args.sampling_strategy
config["evaluation_strategy"] = args.evaluation_strategy
config["cvar_alpha"] = args.cvar_alpha

if config["evaluation_strategy"] == "cvar":
    config["log_name"] = args.log_name+config["sampling_strategy"]+"_alpha_"+str(config["cvar_alpha"])
elif config["evaluation_strategy"] == "mse":
    config["log_name"] = args.log_name + config["sampling_strategy"] + "_GlobalSeed=" \
                             + str(config["global_seed"]) \
                             + "_risklr=" + str(config["risklearner_lr"]) \
                             + "_num_candidates=" + str(config["num_candidates"])


os.system("mkdir -p " + config["log_name"])
config["file_out"] = open(config["log_name"] + "/train_log.txt", "w")

os.system("mkdir -p " + config["log_name"]+"/file")
os.system("cp -r " +" Model/ "+ config["log_name"]+"/file/")
os.system("cp -r " +" Dataset/ "+ config["log_name"]+"/file/")
os.system("cp -r " +" *.py "+ config["log_name"]+"/file/")

os.system("mkdir -p " + config["log_name"]+"/checkpoint")

# print
config["plot_every"] = 100
config["print_every"] = 100
config["test_every"] = 500
config["num_epochs"] = 20000
log_string(config["file_out"], str(config))

# set the task distribution++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
sine_tasks=SineDistribution(0.1, 5, 0, np.pi, -5, 5)

# set the backbone+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
net=SineNet().to(config["device"])

# set the risklearner++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# set the core algorithm+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import datetime
unique_token = "{}__{}__seed{}".format(config['sampling_strategy'],datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), config["global_seed"])
wandb.init(project="sinusoid", config=config, name=unique_token)
maml=SineMAML(sine_tasks, net, config)
maml.outer_loop()

if config["evaluation_strategy"] == "cvar":
    log_string(config['file_out'],
               "Best_epoch: {}, test_cvar_score: {}, test_NSE_score: {}, test_MSE_std: {}".format(maml.best_epoch,
                                                                                                  maml.best_test_cvar,
                                                                                                  maml.best_test_score,
                                                                                                  maml.best_test_std))
elif config["evaluation_strategy"] == "mse":
    log_string(config['file_out'], "Best_epoch: {}, test_MSE_score: {}, test_MSE_std: {}".format(maml.best_epoch,
                                                                                                 maml.best_test_score,
                                                                                                 maml.best_test_std))


# save and print+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
torch.save(maml.meta_losses, config["log_name"]+'/meta_losses.pt')
torch.save(maml.meta_risklearner_losses, config["log_name"]+'/meta_risklearner_losses.pt')
torch.save(maml.test_score_all, config["log_name"]+'/test_score_all.pt')
torch.save(maml.test_std_all, config["log_name"]+'/test_std_all.pt')
torch.save(maml.test_cvar_all, config["log_name"]+'/test_cvar_all.pt')
# plot_line(maml.meta_losses, maml.test_score_all, config["log_name"], config["sampling_strategy"])


import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import pdb

from torch.distributions import Normal
from utils import calcu_cvar

def single_task_test(config, og_net, x, y, axis, task):
    device = config["device"]
    test_alpha = config["test_alpha"]
    test_inner_steps = config["test_inner_steps"]
    criterion = nn.MSELoss()

    # modeling initialization
    dummy_net = nn.Sequential(OrderedDict([
        ('l1', nn.Linear(1, 40)),
        ('relu1', nn.ReLU()),
        ('l2', nn.Linear(40, 40)),
        ('relu2', nn.ReLU()),
        ('l3', nn.Linear(40, 1))]))
    dummy_net = dummy_net.to(device)
    dummy_net.load_state_dict(og_net.state_dict())
    optim = torch.optim.SGD
    opt = optim(dummy_net.parameters(), lr=test_alpha)

    losses = []
    outputs = {}

    # fast adaptation for #inner_steps
    for i in range(test_inner_steps):
        out = dummy_net(x)
        loss = criterion(out, y)
        losses.append(loss.item())

        dummy_net.zero_grad()
        loss.backward()
        opt.step()

    # Inference for the current task
    outputs['initial'] = og_net(axis.view(-1, 1)).clone().detach().cpu().numpy()
    outputs['adapted'] = dummy_net(axis.view(-1, 1)).clone().detach().cpu().numpy()
    outputs["gt"] = task.true_sine(axis.cpu().clone().numpy())

    meta_losses = criterion(torch.tensor(outputs["adapted"]), torch.tensor(outputs["gt"]).unsqueeze(1))
    #     if meta_losses > 100:
    #         print(losses)
    #         pdb.set_trace()
    return outputs, meta_losses

def test(config, epoch, tasks, og_net):
    k = config["k"]
    device = config["device"]
    cvar_alpha = config["cvar_alpha"]

    # +++++++++++++++++++++
    # Fixed Meta-Test Tasks
    seed = 1000000
    np.random.seed(seed)
    num_test_tasks = 1000
    # +++++++++++++++++++++
    test_MSE_all = []

    # Generating test task discriptors
    test_amp_list, test_phase_list = tasks.task_descriptor_candidate(num_test_tasks)
    # print("Here is the test candidate amp" )
    # print(test_amp_list[:10])
    # pdb.set_trace()

    # Testing
    for i in range(num_test_tasks):
        test_task = tasks.active_sample_task(test_amp_list[i], test_phase_list[i])
        # target set
        x, y = test_task.sample_data(k)
        x, y = x.to(device), y.to(device)

        # query set
        axis = np.linspace(-5, 5, 1000)
        axis = torch.tensor(axis, dtype=torch.float)
        axis = axis.to(device)

        outputs, meta_losses = single_task_test(config, og_net, x, y, axis, test_task)
        test_MSE_all.append(meta_losses)
    #         if i==49:
    #             plot_test(x, outputs, axis)
    #             print("Here is the meta_loss of the curren task {} at the {} epoch".format(meta_losses, epoch))
    #         if i in {1, 2}:
    #             print(x)
    #     print(test_amp_list[:10])

    test_MSE_all = torch.Tensor(test_MSE_all)
    # pdb.set_trace()
    test_score = test_MSE_all.mean()
    test_score_std = test_MSE_all.std()
    test_cvar_list = []
    for cvar_alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        test_cvar = torch.tensor(calcu_cvar(test_MSE_all, cvar_alpha))
        test_cvar_list.append(test_cvar)
    # np.random.seed()  # ！！！ very important to release the fixed seed for training.
    np.random.seed(config["global_seed"])
    return test_score, test_score_std, test_cvar_list

def test_pcc(config, epoch, tasks, og_net, risklearner, prior_mu, prior_sigma):
    k = config["k"]
    device = config["device"]

    # +++++++++++++++++++++
    # Fixed Meta-Test Tasks
    seed = 1000000
    np.random.seed(seed)
    # num_test_tasks = 100
    num_test_tasks = 1000
    # +++++++++++++++++++++
    # Generating test task discriptors
    test_amp_list, test_phase_list = tasks.task_descriptor_candidate(num_test_tasks)

    # +++++++++++++++++++++ Risk Prediction
    Risk_X_candidate = torch.cat((torch.tensor(test_amp_list), torch.tensor(test_phase_list)),dim=1)  # Shape: num_candidate * 2
    Risk_X_candidate = Risk_X_candidate.unsqueeze(0).to(device)
    # 1 * 1000 * 2


    z_prior = Normal(prior_mu, prior_sigma)
    num_samples = 20
    z_sample = z_prior.rsample([num_samples])
    # 20 * 1 * 10

    output_type = "deterministic"
    p_y_pred = risklearner.xz_to_y(Risk_X_candidate, z_sample, output_type)
    # 20 * 1000 * 1

    output_mu = torch.mean(p_y_pred, dim=0)
    output_sigma = torch.std(p_y_pred, dim=0)

    test_predicted_MSE_all = [output_mu, output_sigma]

    # +++++++++++++++++++++ Real testing
    test_MSE_all = []
    # Testing
    for i in range(num_test_tasks):
        test_task = tasks.active_sample_task(test_amp_list[i], test_phase_list[i])
        # target set
        x, y = test_task.sample_data(k)
        x, y = x.to(device), y.to(device)

        # query set
        axis = np.linspace(-5, 5, 1000)
        axis = torch.tensor(axis, dtype=torch.float)
        axis = axis.to(device)

        outputs, meta_losses = single_task_test(config, og_net, x, y, axis, test_task)
        test_MSE_all.append(meta_losses)
    #         if i==49:
    #             plot_test(x, outputs, axis)
    #             print("Here is the meta_loss of the curren task {} at the {} epoch".format(meta_losses, epoch))
    #         if i in {1, 2}:
    #             print(x)
    #     print(test_amp_list[:10])

    test_MSE_all = torch.Tensor(test_MSE_all)
    test_score = test_MSE_all.mean()
    test_score_std = test_MSE_all.std()
    np.random.seed()  # ！！！ very important to release the fixed seed for training.
    return test_score, test_score_std, test_MSE_all, test_predicted_MSE_all, Risk_X_candidate

def test_pcc_train(config, epoch, tasks, og_net, risklearner, prior_mu, prior_sigma, observed_x, type):
    k = config["k"]
    device = config["device"]

    # +++++++++++++++++++++
    # Fixed Meta-Test Tasks
    seed = 1000000
    np.random.seed(seed)
    # num_test_tasks = 100
    num_test_tasks = 1000

    if type=="grid":
        # grid++++++++++++++++++++
        grid_amp, grid_phase = np.mgrid[0:5:5 / 10, 0:np.pi:np.pi / 10]
        grid_amp = grid_amp.reshape(-1)
        grid_phase = grid_phase.reshape(-1)
        Risk_X_candidate = torch.cat((torch.tensor(grid_amp).unsqueeze(1), torch.tensor(grid_phase).unsqueeze(1)), dim=1)
        Risk_X_candidate = Risk_X_candidate.unsqueeze(0).to(device)
        Risk_X_candidate = Risk_X_candidate.to(torch.float32)

    elif type=="observe":
        # observation++++++++++++++++++++
        Risk_X_candidate = observed_x
        Risk_X_candidate = Risk_X_candidate.unsqueeze(0).to(device)
        # 1 * 10000 * 2

    z_prior = Normal(prior_mu, prior_sigma)
    num_samples = 20
    z_sample = z_prior.rsample([num_samples])
    # 20 * 1 * 10

    output_type = "deterministic"
    p_y_pred = risklearner.xz_to_y(Risk_X_candidate, z_sample, output_type)
    # 20 * 1000 * 1

    output_mu = torch.mean(p_y_pred, dim=0)
    output_sigma = torch.std(p_y_pred, dim=0)

    test_predicted_MSE_all = [output_mu, output_sigma]

    # +++++++++++++++++++++ Real testing
    test_MSE_all = []
    # Testing
    for i in range(Risk_X_candidate.shape[1]):
        # test_task = tasks.active_sample_task(observed_x[i, 0].cpu(), observed_x[i, 1].cpu())
        test_task = tasks.active_sample_task(Risk_X_candidate[0, i, 0].cpu(), Risk_X_candidate[0, i, 1].cpu())
        # target set
        x, y = test_task.sample_data(k)
        x, y = x.to(device), y.to(device)

        # query set
        axis = np.linspace(-5, 5, 1000)
        axis = torch.tensor(axis, dtype=torch.float)
        axis = axis.to(device)

        outputs, meta_losses = single_task_test(config, og_net, x, y, axis, test_task)
        test_MSE_all.append(meta_losses)
    #         if i==49:
    #             plot_test(x, outputs, axis)
    #             print("Here is the meta_loss of the curren task {} at the {} epoch".format(meta_losses, epoch))
    #         if i in {1, 2}:
    #             print(x)
    #     print(test_amp_list[:10])

    test_MSE_all = torch.Tensor(test_MSE_all)
    test_score = test_MSE_all.mean()
    test_score_std = test_MSE_all.std()
    return test_score, test_score_std, test_MSE_all, test_predicted_MSE_all, Risk_X_candidate



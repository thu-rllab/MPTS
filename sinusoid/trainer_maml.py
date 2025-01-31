import torch
import torch.nn as nn
import time
import numpy as np
from Model.trainer_risklearner import RiskLearnerTrainer
import pdb
from utils import log_string, pearson_correlation_coefficient_torch
from test import test
import wandb
import matplotlib.pyplot as plt
from Model.risklearner import RiskLearner

class SineMAML():
    def __init__(self, sine_tasks, net, config):
        self.config = config
        self.tasks = sine_tasks
        self.net = net
        self.risklearner = RiskLearner(x_dim=2, y_dim=1, r_dim=10, z_dim=10, h_dim=10).to(config["device"])
        self.device = config["device"]

        # maml hyparpameters
        self.weights = list(net.parameters())
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.k = config["k"]
        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, self.beta)

        # Sampling
        self.num_metatasks = config["num_metatasks"]
        self.num_candidates = config["num_candidates"]
        self.sampling_strategy = config["sampling_strategy"]
        self.risklearner_lr = config["risklearner_lr"]
        risklearner_optimizer = torch.optim.Adam(self.risklearner.parameters(), lr=self.risklearner_lr)
        if "mpts" in self.sampling_strategy:
            self.risklearner_trainer = RiskLearnerTrainer(self.sampling_strategy, self.device, self.risklearner, risklearner_optimizer, self.num_metatasks)

        self.gamma_mu = config["gamma_mu"]
        self.gamma_sigma = config["gamma_sigma"]

        # print
        self.plot_every = config["plot_every"]
        self.print_every = config["print_every"]
        self.test_every = config["test_every"]
        self.num_epochs = config["num_epochs"]
        self.meta_losses = []
        self.meta_risklearner_losses = []
        self.test_score_all = []
        self.test_std_all = []
        self.test_cvar_all = []

        self.file_out = config["file_out"]
        self.best_epoch = 0
        self.best_test_score = 100
        self.best_test_cvar = 100

        if self.sampling_strategy == "group_dro":
            self.adv_probs = torch.ones(self.num_metatasks).to(self.device)
            self.robust_step_size = 0.001


    def inner_loop(self, task):
        temp_weights = [w.clone() for w in self.weights]
        # Fast adaptation on the support set
        x, y = task.sample_data(size=self.k)  # sampling D
        x, y = x.to(self.device), y.to(self.device)
        output = self.net.argforward(x, temp_weights)
        loss = self.criterion(output, y)
        grads = torch.autograd.grad(loss, temp_weights)
        temp_weights = [w - self.alpha * g for w, g in zip(temp_weights, grads)]  # temporary update of weights

        # Testing on the query set
        x, y = task.sample_data(size=self.k)  # sampling D'
        x, y = x.to(self.device), y.to(self.device)
        output = self.net.argforward(x, temp_weights)
        metaloss = self.criterion(output, y)
        return metaloss

    def outer_loop(self):
        total_loss = 0.0
        total_pcc = 0.0
        total_risklearner_loss = 0.0
        train_start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):


            metaloss_sum = 0
            metaloss_list = []
            self.meta_optimiser.zero_grad()

            ###################################################################
            # +++++++++++++++++++++
            np.random.seed(epoch)
            # +++++++++++++++++++++
            amp_list_candidate, phase_list_candidate = self.tasks.task_descriptor_candidate(self.num_candidates)

            if self.sampling_strategy == "erm":
                amp_list, phase_list = amp_list_candidate[:self.num_metatasks], phase_list_candidate[:self.num_metatasks]

                # innerloop: dealing with each task one by one.+++++++++++++++++++
                for i in range(self.num_metatasks):
                    task = self.tasks.active_sample_task(amp_list[i], phase_list[i])
                    metaloss = self.inner_loop(task) # query loss on the task
                    metaloss_sum += metaloss
                    metaloss_list.append(metaloss)
                risklearner_loss = metaloss * 0.0

            elif self.sampling_strategy == "drm":

                # innerloop: dealing with each task one by one.+++++++++++++++++++
                for i in range(self.num_candidates):
                    task = self.tasks.active_sample_task(amp_list_candidate[i], phase_list_candidate[i])
                    metaloss = self.inner_loop(task)
                    # metaloss_sum += metaloss
                    metaloss_list.append(metaloss)

                metaloss_list.sort()
                metaloss_sum = sum(metaloss_list[-self.num_metatasks:]) # worst_case !!!!
                risklearner_loss = metaloss * 0.0

            elif self.sampling_strategy == "gdrm":
                amp_list, phase_list = amp_list_candidate[:self.num_metatasks], phase_list_candidate[:self.num_metatasks]
                # innerloop: dealing with each task one by one.+++++++++++++++++++
                for i in range(self.num_metatasks):
                    task = self.tasks.active_sample_task(amp_list[i], phase_list[i])
                    metaloss = self.inner_loop(task)
                    # metaloss_sum += metaloss
                    metaloss_list.append(metaloss)

                # pdb.set_trace()
                metaloss_tensor = torch.stack(metaloss_list)
                # metaloss_list = torch.tensor(metaloss_list)
                self.adv_probs = self.adv_probs * torch.exp(self.robust_step_size * metaloss_tensor.data)
                self.adv_probs = self.adv_probs / (self.adv_probs.sum()) * self.num_metatasks
                metaloss_tensor = metaloss_tensor @ self.adv_probs
                metaloss_sum = metaloss_tensor.sum()

                risklearner_loss = metaloss * 0.0

            elif self.sampling_strategy == "mpts":

                Risk_X_candidate = torch.cat((torch.tensor(amp_list_candidate),torch.tensor(phase_list_candidate)), dim=1) # Shape: num_candidate * 2
                acquisition_score, p_y_pred = self.risklearner_trainer.acquisition_function(Risk_X_candidate, self.gamma_mu, self.gamma_sigma)
                acquisition_score = acquisition_score.squeeze(1) # Shape: num_candidate

                _, selected_index = torch.topk(acquisition_score, k=self.num_candidates)
                start_point = 0
                selected_index = selected_index[start_point:start_point + self.num_metatasks]
                selected_index = selected_index.cpu()

                amp_list = amp_list_candidate[selected_index]
                phase_list = phase_list_candidate[selected_index]

                # innerloop: dealing with each task one by one.+++++++++++++++++++
                for i in range(self.num_metatasks):
                    task = self.tasks.active_sample_task(amp_list[i], phase_list[i])
                    metaloss = self.inner_loop(task)
                    metaloss_sum += metaloss
                    metaloss_list.append(metaloss)

                # Training Sampling Module +++++++++++++++++++++++++++++++++++++++
                Risk_X = torch.cat((torch.tensor(amp_list), torch.tensor(phase_list)), dim=1)
                Risk_Y = torch.tensor(metaloss_list) / self.k
                Risk_X, Risk_Y = Risk_X.to(self.device), Risk_Y.to(self.device)
                # Shape: num_metatasks * 2
                # Shape: num_metatasks
                risklearner_loss = self.risklearner_trainer.train(Risk_X, Risk_Y)

                predicted_risk = p_y_pred.mean(0).squeeze(1)[selected_index] # 32
                pcc = pearson_correlation_coefficient_torch(Risk_Y, predicted_risk)

            np.random.seed(self.config["global_seed"])
            ########################################################################

            '''shared by all sampling strategies!!!!!!!!!!!!!!!!!'''
            '''WITH metaloss_sum'''
            # important step: update the backbone parameters in the last step
            metagrads = torch.autograd.grad(metaloss_sum, self.weights)
            for w, g in zip(self.weights, metagrads):
                w.grad = g
            self.meta_optimiser.step()

            # printing
            total_loss += metaloss_sum.item() / self.num_metatasks
            total_risklearner_loss += risklearner_loss.item()

            if "mpts" in self.sampling_strategy:
                total_pcc += pcc.item()



            if epoch % self.print_every == 0:
                train_end_time = time.time()
                # log_string(self.file_out, "{}/{}. training_time for {} epochs:{}, loss: {}, risklearner_loss: {}".format(epoch, self.num_epochs, self.print_every, (train_end_time - train_start_time), total_loss / self.print_every, total_risklearner_loss/self.print_every))
                log_string(self.file_out,
                           "{}/{}. training_time for {} epochs:{}, loss: {}, risklearner_loss: {}, pcc:{}".format(epoch,
                                                                                                          self.num_epochs,
                                                                                                          self.print_every,
                                                                                                          (train_end_time - train_start_time),
                                                                                                          total_loss / self.print_every,
                                                                                                          total_risklearner_loss / self.print_every,
                                                                                                          total_pcc / self.print_every))
                
                wandb.log({"meta_loss": total_loss / self.plot_every, 
                           "risklearner_loss": total_risklearner_loss / self.plot_every, 
                            "pcc": total_pcc / self.print_every,
                           "epoch": epoch})
                train_start_time = time.time()

            if epoch % self.plot_every == 0:
                self.meta_losses.append(total_loss / self.plot_every)
                self.meta_risklearner_losses.append(total_risklearner_loss / self.plot_every)
                total_loss = 0
                total_pcc = 0
                total_risklearner_loss = 0

            # Testing
            if epoch % self.test_every == 0:
                test_start_time = time.time()
                test_score, test_score_std, test_cvar_list = test(self.config, epoch, self.tasks, self.net.net)
                wandb.log({"test_score": test_score, "test_score_std": test_score_std, 
                           "test_score_cvar_0_1": test_cvar_list[0], 
                           "test_score_cvar_0_3": test_cvar_list[1], 
                           "test_score_cvar_0_5": test_cvar_list[2],
                           "test_score_cvar_0_7": test_cvar_list[3],
                           "test_score_cvar_0_9": test_cvar_list[4],
                           "epoch": epoch})
                test_end_time = time.time()
                # print("{}/{}. test_MSE: {}, std: {}".format(epoch, self.num_epochs, test_score, test_score_std))
                test_cvar_list = torch.stack(test_cvar_list)
                log_string(self.file_out, "{}/{}. test_time:{} test_MSE: {}, std: {}, test_cvar_list: {}".format(epoch, self.num_epochs, (test_end_time - test_start_time), test_score, test_score_std, test_cvar_list))

                self.test_score_all.append(test_score)
                self.test_std_all.append(test_score_std)
                self.test_cvar_all.append(test_cvar_list)

                if self.config["evaluation_strategy"] == "mse":
                    if test_score < self.best_test_score:
                        self.best_epoch = epoch
                        self.best_test_score = test_score
                        self.best_test_std = test_score_std
                        torch.save({
                                    'best_epoch': self.best_epoch,
                                    'best_test_score': self.best_test_score,
                                    'test_score_std': self.best_test_std,
                                    'model_state_dict': self.net.net.state_dict(),
                                    "risklearner_state_dict": self.risklearner.state_dict(),
                                    "z_prior_loc": self.risklearner_trainer.z_prior.loc if "mpts" in self.sampling_strategy else 0,
                                    "z_prior_scale": self.risklearner_trainer.z_prior.scale if "mpts" in self.sampling_strategy else 0,
                                    },
                                    self.config["log_name"]+"/checkpoint/Best_model")

                elif self.config["evaluation_strategy"] == "cvar":
                    if test_cvar_list[2] < self.best_test_cvar:
                        self.best_epoch = epoch
                        self.best_test_score = test_score
                        self.best_test_std = test_score_std
                        self.best_test_cvar = test_cvar_list[2]
                        torch.save({
                                    'best_epoch': self.best_epoch,
                                    'best_test_cvar': self.best_test_cvar,
                                    'best_test_score': self.best_test_score,
                                    'test_score_std': self.best_test_std,
                                    'model_state_dict': self.net.net.state_dict(),
                                    "risklearner_state_dict": self.risklearner.state_dict(),
                                    "z_prior_loc": self.risklearner_trainer.z_prior.loc if "mpts" in self.sampling_strategy else 0,
                                    "z_prior_scale": self.risklearner_trainer.z_prior.scale if "mpts" in self.sampling_strategy else 0,
                                    },
                                    self.config["log_name"]+"/checkpoint/Best_cvar_model")

                train_start_time = time.time()

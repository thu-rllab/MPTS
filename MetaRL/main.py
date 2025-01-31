import datetime
import os
import copy
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy.stats as st
import torch
import wandb
import utils
from arguments import parse_args

from baseline import LinearFeatureBaseline
from metalearner import MetaLearner
from policies.categorical_mlp import CategoricalMLPPolicy
from policies.normal_mlp import NormalMLPPolicy
from sampler import BatchSampler, MP_BatchSampler


###################################################################################
    # rewards computation and statistics functions
###################################################################################

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

    return returns

def get_returns(episodes_per_task):

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
            ret = (episodes[update_idx].rewards * episodes[update_idx].mask).sum(dim=0)
            curr_returns.append(ret)
        # for training, curr_returns [training episodes rewards, fast adaptation episodes rewards], list shape [2], training episodes rewards with shape [num_eval_traj]
        # for testing, curr_returns [initial rewards,  fast adaptation episodes rewards of num_test_step updates], list shape [1+num_test_steps], episodes rewards with shape [num_eval_traj]

        # for training, result of torch.stack(curr_returns, dim=1) will be: [num_eval_traj, 2], here 2 means the training/validation
        # for testing, result of torch.stack(curr_returns, dim=1) will be: [num_eval_traj, num_test_steps+1]
        returns.append(torch.stack(curr_returns, dim=1))

    returns = torch.stack(returns) # for training, returns shape [num_tasks, num_eval_traj, 2]; for testing, returns shape [num_tasks, num_eval_traj, num_test_steps+1]
    returns = returns.reshape((-1, returns.shape[-1])) # for training, returns shape [num_tasks*num_eval_traj, 2]; for testing, returns shape [num_tasks*num_eval_traj, num_test_steps+1]

    return returns


def total_rewards(episodes_per_task, interval=False):

    returns = get_returns(episodes_per_task).cpu().numpy() # returns shape [num_tasks*num_eval_traj, 2] (or [num_tasks*num_eval_traj, num_test_steps+1])

    mean = np.mean(returns, axis=0) 
    # for training, mean [meta training rewards averaged over task batch, eval adaptation rewards averaged over task batch], shape (2,)
    # for testing, mean [num_test_steps+1 adaptation rewards averaged over task batch], shape(num_test_steps+1,)

    # record the cvar returns of fast adaptation on the last updates
    alpha = [0.9, 0.7, 0.5, 0.95]
    cvar_adapt_returns = []
    for i in alpha:
        cvar_ind = returns[:,-1].argsort()[:int((1-i)*returns.shape[0])]
        cvar_ind_list = cvar_ind.tolist()
        cvar_adapt_returns.append(np.mean(returns[cvar_ind_list,-1])) 

    cvar_adapt_returns_step1 = []
    for i in alpha:
        cvar_ind = returns[:,1].argsort()[:int((1-i)*returns.shape[0])]
        cvar_ind_list = cvar_ind.tolist()
        cvar_adapt_returns_step1.append(np.mean(returns[cvar_ind_list,1])) 

    conf_int = st.t.interval(0.95, len(mean) - 1, loc=mean, scale=st.sem(returns, axis=0))
    conf_int = mean - conf_int
    if interval:
        # return a tuple (mean rewards, confidence)
        return mean, conf_int[0]
    else:
        return mean, cvar_adapt_returns, cvar_adapt_returns_step1


###################################################################################
    # main function as follows
###################################################################################
    
    
def main(args):
    print('starting....')
    print('game_framework', args.game_framework)
    print('init_dist', args.init_dist)
    raw_framework_name = {1: 'DR_MAML', 2: 'Vanilla_MAML', 3: 'GroupDRO_MAML'}
    alg_name = raw_framework_name[args.game_framework]
    if args.game_framework == 2:
        if args.sampling_strategy == 'mp':
            alg_name = 'task_sample_MAML'
    args.unique_token = "{}__{}__{}__{}__{}__seed{}".format(args.game_framework, alg_name, args.task,args.env_name,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.seed)
    if not args.no_wandb_or_modelsave:
        wandb.init(project=args.wandb_project,name=args.unique_token, config=vars(args))

    if args.game_framework == 1:
        args.baseline_use_last_r = True
        args.meta_batch_size = args.meta_batch_size * 2
    elif args.game_framework == 3:
        args.baseline_use_last_r = False

    utils.set_seed(args.seed, cudnn=args.make_deterministic)
    
    # here -v0 means the vanilla envs, while -v1 means the adversarially robust envs
    continuous_actions = (args.env_name in ['HalfCheetahVel-v0', 
                                            'HalfCheetahMassVel-v1',    
                                            'Walker2dMassVel-v1',
                                            'Walker2dVel-v1',
                                            'ReacherPos-v0',
                                            ])  
    n_params = {'HalfCheetahVel-v0':1, 'HalfCheetahMassVel-v1':2, 
                'Walker2dMassVel-v1':2, 'Walker2dVel-v1':1, 'ReacherPos-v0':2}
    
    iter_inner_loss = []
    iter_outer_loss = []

    init_cvar_after_reward_10 = []
    init_cvar_after_reward_30 = []
    init_cvar_after_reward_50 = []
    
    # eval in initial task dist: keep track of the rewards of tasks before fast adaptation
    init_dist_before_reward = []
    # eval in initial task dist: keep track of the rewards of tasks after fast adaptation 
    init_dist_after_reward = []
    
    # track eval returns after multi-step fast updates
    # [arr[reward_test_step0, ...,reward_num_test_steps]], list len num_test_steps+1
    init_multi_steps_reward = []
    init_multi_steps_cvar_reward_10 = []
    init_multi_steps_cvar_reward_30 = []
    init_multi_steps_cvar_reward_50 = []
        
    # this is the prefix of storing the model
    exp_string_outer = args.unique_token
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##################################################################################################################
    # to introduce Meta Learner --> Optimize policies
    ##################################################################################################################

    sampler = BatchSampler(args.env_name, 
                           batch_size=args.fast_batch_size, 
                           num_workers=max(args.test_batch_size, args.fast_batch_size),
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
        
    #### !!!update sampling strategy############
    if args.sampling_strategy != 'random':
        from MPModel.risklearner import RiskLearner
        from MPModel.new_trainer_risklearner import RiskLearnerTrainer
        risklearner = RiskLearner(n_params[args.env_name], 1, 10, 10, 10).to(device)
        risklearner_optimizer = torch.optim.Adam(risklearner.parameters(), lr=args.sampler_lr)
        
        risklearner_trainer = RiskLearnerTrainer(device, risklearner, risklearner_optimizer, output_type=args.output_type, kl_weight=args.kl_weight)

        sampler = MP_BatchSampler(args, risklearner_trainer, 
                                args.sampling_gamma_0,
                                args.sampling_gamma_1,
                                args.env_name, 
                        batch_size=args.fast_batch_size, 
                        num_workers=max(args.test_batch_size, args.fast_batch_size),
                        device=args.device, seed=args.seed)

    #################################

    

    # initialise baseline
    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))

    # initialise meta-learner
    metalearner = MetaLearner(args,
                              sampler, 
                              policy, 
                              baseline, 
                              gamma=args.gamma, 
                              fast_lr=args.fast_lr, 
                              tau=args.tau,
                              device=args.device)        

    for i_iter in range(args.num_batches):

        tasks = sampler.sample_tasks(i_iter, args.meta_batch_size, 
                                    args.init_dist)

        # do the inner-loop update for each task
        # this returns training (before update) and validation (after update/fast adaptation resulting) episodes
        # episodes: list of tuples [(training episodes, validation episodes)], shape [num_tasks], new samples from the adapted policies' interaction
        episodes, inner_losses, after_inner_losses = metalearner.sample(tasks, 
                                                    first_order=args.first_order)

        

        # take the meta-gradient step
        if args.game_framework == 1:
            # distributionally robust maml, manually set cvar_conf, e.g. 0.7
            outer_loss, batch_outer_loss = metalearner.step(episodes, 
                                                            max_kl=args.max_kl, 
                                                            cg_iters=args.cg_iters,
                                                            cg_damping=args.cg_damping, 
                                                            ls_max_steps=args.ls_max_steps,
                                                            ls_backtrack_ratio=args.ls_backtrack_ratio, 
                                                            mean_loss=False, 
                                                            cvar_conf=0.5,
                                                            game_framework=1)
        elif args.game_framework == 2:
            # vanilla maml, assert cvar_conf = 0.0
            outer_loss = metalearner.step(episodes, 
                                          max_kl=args.max_kl, 
                                          cg_iters=args.cg_iters,
                                          cg_damping=args.cg_damping, 
                                          ls_max_steps=args.ls_max_steps,
                                          ls_backtrack_ratio=args.ls_backtrack_ratio, 
                                          mean_loss=True, 
                                          cvar_conf=0.0,
                                          game_framework=2)
        elif args.game_framework == 3:
            # group dro maml, assert cvar_conf = 0.0
            outer_loss, batch_outer_loss = metalearner.step(episodes, 
                                                            max_kl=args.max_kl, 
                                                            cg_iters=args.cg_iters,
                                                            cg_damping=args.cg_damping, 
                                                            ls_max_steps=args.ls_max_steps,
                                                            ls_backtrack_ratio=args.ls_backtrack_ratio, 
                                                            mean_loss=False, 
                                                            cvar_conf=0.0,
                                                            game_framework=3)

        
        iter_inner_loss.append(np.mean(inner_losses))
        iter_outer_loss.append(outer_loss.item())

        curr_returns, cvar_adapt_returns,_ = total_rewards(episodes, interval=False) # return arrary [task avg training rewards, task avg eval adapt rewards], shape [2]
        print (i_iter)
        print('running_adaptation_returns_on_updated_adv_dist: ', curr_returns[1])

        #### !!!update sampling strategy############
        if args.sampling_strategy != 'random':
            inner_returns = get_task_returns(episodes, use_last_r = args.use_last_r)#n_tasks, n_traj,train/val
            mean_inner_after_returns = -torch.mean(inner_returns[:,:,-1], dim=1, keepdim=True)
            acquisition_score, acquisition_mean, acquisition_std = sampler.get_acquisition_score(i_iter, tasks)#ntask,1
            if not args.no_wandb_or_modelsave:
                wandb.log({'acquisition_mean':acquisition_mean.mean().item(),
                            'acquisition_std':acquisition_std.mean().item(),
                    'return_corr':np.corrcoef(acquisition_score.squeeze().cpu().detach().numpy(), mean_inner_after_returns.squeeze().cpu().detach().numpy())[0,1],  
                        'iter': i_iter,
                        't':sampler.total_t})

            inner_returns = - inner_returns[:,:,-1]#n_tasks, n_traj
            return_norm=loss_norm=1.0
            if not args.batch_norm:
                return_norm = 300.0 if not args.use_last_r else 1.0
                loss_norm = 0.01
            if args.use_inner_loss:
                y = torch.tensor(after_inner_losses).to(device)/loss_norm
            else:
                y = inner_returns.to(device)/return_norm
            if args.batch_norm:
                y = (y - y.mean()) / (y.std() + 1e-8)
                
            for _ in range(args.sampler_train_times):
                sampler_loss, recon_loss, kl_loss = sampler.train(tasks, y)
            if not args.no_wandb_or_modelsave:
                wandb.log({'inner_returns':torch.mean(inner_returns).item(),
                        'after_inner_loss':np.mean(after_inner_losses),
                        'y':y.mean().item(),
                        'y_std':y.std().item(),
                        'sampler_loss': sampler_loss, 
                        'recon_loss': recon_loss,
                        'kl_loss':kl_loss,  
                        'iter': i_iter,
                        't':sampler.total_t})
        else:
            if not args.no_wandb_or_modelsave:
                wandb.log({'iter': i_iter,
                        't':sampler.total_t})
        ############################################
        
        # to record the support episode rewards
        init_dist_before_reward.append(curr_returns[0])
        # to record the query episode rewards (for fast adaptation evaluation)
        init_dist_after_reward.append(curr_returns[1])            
        
        init_cvar_after_reward_10.append(cvar_adapt_returns[0])
        init_cvar_after_reward_30.append(cvar_adapt_returns[1])
        init_cvar_after_reward_50.append(cvar_adapt_returns[2])
        
        # -- evaluation

        # evaluate for multiple update steps
        if i_iter % args.test_freq == 0 or i_iter == args.num_batches - 1:
            init_multi_steps1_cvar_reward_5 = []
            init_multi_steps1_cvar_reward_10 = []
            init_multi_steps1_cvar_reward_30 = []
            init_multi_steps1_cvar_reward_50 = []
            # evaluate the test task returns from initial task dist
            # this applies to vanilla-maml, CVaR-maml and ar-maml
            eval_init_returns,  eval_init_cvar_returns, eval_init_cvar_returns_step1= eval_iter(args, 
                                                                  sampler, 
                                                                  metalearner, 
                                                                  i_iter,
                                                                  n_tasks=args.eval_meta_batch_size)
            init_multi_steps_reward.append(eval_init_returns)
            init_multi_steps_cvar_reward_10.append(eval_init_cvar_returns[0])
            init_multi_steps_cvar_reward_30.append(eval_init_cvar_returns[1])
            init_multi_steps_cvar_reward_50.append(eval_init_cvar_returns[2])
            init_multi_steps1_cvar_reward_5.append(eval_init_cvar_returns_step1[3])
            init_multi_steps1_cvar_reward_10.append(eval_init_cvar_returns_step1[0])
            init_multi_steps1_cvar_reward_30.append(eval_init_cvar_returns_step1[1])
            init_multi_steps1_cvar_reward_50.append(eval_init_cvar_returns_step1[2])

            # evaluate the test task returens from the adversarial task dist
            # this applies to ar-maml only
            
            if not args.no_wandb_or_modelsave:
                wandb.log({'iter_inner_loss': iter_inner_loss[-1],'iter_outer_loss': iter_outer_loss[-1],
                        'init_dist_before_reward': init_dist_before_reward[-1],
                            'init_dist_after_reward': init_dist_after_reward[-1],
                                'init_cvar_after_reward_10': init_cvar_after_reward_10[-1],
                                'init_cvar_after_reward_30': init_cvar_after_reward_30[-1],
                                'init_cvar_after_reward_50': init_cvar_after_reward_50[-1],
                        
                            'init_multi_steps_cvar_reward_10': init_multi_steps_cvar_reward_10[-1],
                            'init_multi_steps_cvar_reward_30': init_multi_steps_cvar_reward_30[-1],
                            'init_multi_steps_cvar_reward_50': init_multi_steps_cvar_reward_50[-1],
                            'init_multi_steps1_cvar_reward_5': init_multi_steps1_cvar_reward_5[-1],
                            'init_multi_steps1_cvar_reward_10': init_multi_steps1_cvar_reward_10[-1],
                                'init_multi_steps1_cvar_reward_30': init_multi_steps1_cvar_reward_30[-1],
                                'init_multi_steps1_cvar_reward_50': init_multi_steps1_cvar_reward_50[-1],
                            'iter': i_iter,'t':sampler.total_t})
                for i in range(len(init_multi_steps_reward[-1])):
                    wandb.log({'init_multi_steps_reward_'+str(i): init_multi_steps_reward[-1][i], 'iter': i_iter})
 
        # -- save policy network and task dist network
        if not args.no_wandb_or_modelsave:
            if (i_iter % 10 == 0) or (i_iter >= args.num_batches - 20):
                os.makedirs(os.path.join(args.save_model_dir,args.env_name,exp_string_outer), exist_ok=True)
                torch.save(copy.copy(policy.state_dict()), os.path.join(args.save_model_dir,args.env_name,exp_string_outer,'policy-{0}.pt'.format(i_iter)))

    ###################################################################################
        # save the meta training and testing results
    ###################################################################################
        
    

def eval_iter(args, 
              sampler,  
              metalearner, 
              i_iter,
              n_tasks=500):
    '''
    This part is to evaluate the performance of meta learners from the frozen distributiona adversary models.
    '''

    test_tasks = sampler.sample_tasks(i_iter, n_tasks, args.init_dist, test=True)

    # test_episodes in the shape [XX, XX], need to be specified  
    # list len  n_tasks
    # for each emelment in the list: [num_test_steps+1 episodes]
    test_episodes = metalearner.test(test_tasks, 
                                     num_steps=args.num_test_steps,
                                     batch_size=args.test_batch_size, 
                                     halve_lr=args.halve_test_lr)
    all_returns, cvar_reward, cvar_reward_step1 = total_rewards(test_episodes, interval=False)
    if args.sampling_strategy != 'random':
        inner_returns = get_task_returns(test_episodes, use_last_r = args.use_last_r)
        mean_inner_after_returns = -torch.mean(inner_returns[:,:,1], dim=1, keepdim=True)
        acquisition_score, acquisition_mean, acquisition_std = sampler.get_acquisition_score(i_iter, test_tasks)#ntask,1

        if not args.no_wandb_or_modelsave:
            wandb.log({'test_acquisition_mean':acquisition_mean.mean().item(),
                        'test_acquisition_std':acquisition_std.mean().item(),
                'test_return_corr':np.corrcoef(acquisition_score.squeeze().cpu().detach().numpy(), mean_inner_after_returns.squeeze().cpu().detach().numpy())[0,1],  
                    'iter': i_iter,
                    't':sampler.total_t})
    
    return all_returns, cvar_reward, cvar_reward_step1
    


if __name__ == '__main__':
    args = parse_args()

    main(args)
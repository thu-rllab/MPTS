import gym
import numpy as np
import logging

import torch
from common.envs.randomized_vecenv import make_vec_envs

from common.svpg.svpg import SVPG

from common.utils.rollout_evaluation import evaluate_policy, check_solved, cvar_evaluate_policy, gdro_evaluate_policy
from common.agents.ddpg.replay_buffer import ReplayBuffer
from common.sampler.sampler import MP_BatchSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
import wandb

class SVPGSimulatorAgent(object):
    """Simulation object which creates randomized environments based on specified params, 
    handles SVPG-based policy search to create envs, 
    and evaluates controller policies in those environments
    """

    def __init__(self,
                 reference_env_id,
                 randomized_env_id,
                 randomized_eval_env_id,
                 agent_name,
                 nagents,
                 nparams,
                 temperature,
                 svpg_rollout_length,
                 svpg_horizon,
                 max_step_length,
                 reward_scale,
                 initial_svpg_steps,
                 max_env_timesteps,
                 episodes_per_instance,
                 discrete_svpg,
                 load_discriminator,
                 freeze_discriminator,
                 freeze_agent,
                 seed,
                 args,
                #  writer,
                 train_svpg=True,
                 particle_path="",
                 discriminator_batchsz=320,
                 randomized_eval_episodes=3,
                 ):

        # TODO: Weird bug
        assert nagents > 2

        self.reference_env_id = reference_env_id
        self.randomized_env_id = randomized_env_id
        self.randomized_eval_env_id = randomized_eval_env_id
        self.agent_name = agent_name

        self.log_distances = reference_env_id.find('Lunar') == -1

        self.randomized_eval_episodes = randomized_eval_episodes
        if reference_env_id.find('Pusher') != -1:
            self.randomized_eval_episodes = 10
        elif reference_env_id.find('Lunar') != -1:
            self.randomized_eval_episodes = 10
        elif reference_env_id.find('Ergo') != -1:
            self.randomized_eval_episodes = 10

        # Vectorized environments - step with nagents in parallel
        self.reference_env = make_vec_envs(reference_env_id, seed, nagents)
        self.randomized_env = make_vec_envs(randomized_env_id, seed, nagents)
        self.randomized_eval_env = make_vec_envs(randomized_eval_env_id, seed, nagents)

        self.state_dim = self.reference_env.observation_space.shape[0]
        self.action_dim = self.reference_env.action_space.shape[0]
    
        if reference_env_id.find('Pusher') != -1:
            self.hard_env = make_vec_envs('Pusher3DOFHard-v0', seed, nagents)
        elif reference_env_id.find('Lunar') != -1:
            self.hard_env = make_vec_envs('LunarLander10-v0', seed, nagents)
        elif reference_env_id.find('Backlash') != -1:
            self.hard_env = make_vec_envs('ErgoReacherRandomizedBacklashHard-v0', seed, nagents)
        else:
            self.hard_env = make_vec_envs('ErgoReacher4DOFRandomizedHard-v0', seed, nagents)

        self.sampled_regions = [[] for _ in range(nparams)]

        self.nagents = nagents # num parallel envs
        self.nparams = self.randomized_env.randomization_space.shape[0] # dimensions of task identifiers
        assert self.nparams == nparams, "Double check number of parameters: Args: {}, Env: {}".format(
            nparams, self.nparams)

        self.svpg_horizon = svpg_horizon
        self.initial_svpg_steps = initial_svpg_steps
        self.max_env_timesteps = max_env_timesteps
        self.episodes_per_instance = episodes_per_instance
        self.discrete_svpg = discrete_svpg

        self.freeze_discriminator = freeze_discriminator
        self.freeze_agent = freeze_agent

        self.train_svpg = train_svpg
        
        self.agent_eval_frequency = max_env_timesteps * nagents 
        if self.log_distances:
            self.agent_eval_frequency = 25000
        else:
            self.agent_eval_frequency = 50000

        self.seed = seed
        self.svpg_timesteps = 0
        self.agent_timesteps = 0
        self.agent_timesteps_since_eval = 0
        
        from mpmodel.risklearner import RiskLearner
        from mpmodel.new_trainer_risklearner import RiskLearnerTrainer
        self.risklearner = RiskLearner(self.nparams, 1, 10, 10, 10).to(device)
        self.risklearner_optimizer = torch.optim.Adam(self.risklearner.parameters(), lr=args.sampler_lr)
        self.risklearner_trainer = RiskLearnerTrainer(device, self.risklearner, self.risklearner_optimizer, output_type=args.output_type, kl_weight=args.kl_weight)

        self.sampler = MP_BatchSampler(args, self.risklearner_trainer, 
                            args.sampling_gamma_0,
                            args.sampling_gamma_1,)

        self.args = args
        if not self.freeze_agent:
            self.replay_buffer = ReplayBuffer()
        else:
            self.replay_buffer = None

        self.svpg = SVPG(nagents=nagents,
                         nparams=self.nparams,
                         max_step_length=max_step_length,
                         svpg_rollout_length=svpg_rollout_length,
                         svpg_horizon=svpg_horizon,
                         temperature=temperature,
                         discrete=self.discrete_svpg,
                         kld_coefficient=0.0)

        if particle_path != "":
            logger.info("Loading particles from: {}".format(particle_path))
            self.svpg.load(directory=particle_path)


    def select_action(self, agent_policy):
        """Select an action based on SVPG policy, where an action is the delta in each dimension.
        Update the counts and statistics after training agent,
        rolling out policies, and calculating simulator reward.
        """
        if self.agent_timesteps >= self.args.uniform_sample_steps * self.args.max_agent_timesteps:
            # Get sim instances from SVPG policy
            # simulation_instances = self.svpg.step()

            # index = self.svpg_timesteps % self.svpg_horizon
            # self.simulation_instances_full_horizon[:, index, :, :] = simulation_instances
            if 'mpts' in self.args.algo:
                simulation_instances = self.sampler.sample_tasks(shape=(self.nagents,self.svpg.svpg_rollout_length,self.svpg.nparams),multiplier=self.args.sampler_multiplier)
        else:
            simulation_instances = np.random.uniform(0.0, 1.0, size=(self.nagents,
                                self.svpg.svpg_rollout_length,
                                self.svpg.nparams))

        assert (self.nagents, self.svpg.svpg_rollout_length, self.svpg.nparams) == simulation_instances.shape

        # Create placeholders for trajectories
        randomized_returns = []
        randomized_dists = []


        # Reshape to work with vectorized environments
        simulation_instances = np.transpose(simulation_instances, (1, 0, 2)) # (5,10,1) (self.svpg.svpg_rollout_length,self.nagents,self.svpg.nparams)

        # Create environment instances with vectorized env, and rollout agent_policy in both
        for t in range(self.svpg.svpg_rollout_length):
            logging.info('Iteration t: {}/{}'.format(t, self.svpg.svpg_rollout_length))  

            # reference_trajectory, reference_return = self.rollout_agent(agent_policy)

            self.randomized_env.randomize(randomized_values=simulation_instances[t])
            randomized_trajectory, randomized_return, randomized_dist = self.rollout_agent(agent_policy, reference=False, cvar=self.args.cvar if self.args.algo=='drm' else None, gdroweight=self.args.gdroweight if self.args.algo=='gdrm' else None) 

            # reference_returns.append(reference_return)
            randomized_returns.append(randomized_return)
            randomized_dists.append(randomized_dist)

            for i in range(self.nagents if self.args.algo != 'drm' else int((1 - self.args.cvar) * self.nagents)):
                self.agent_timesteps += len(randomized_trajectory[i])
                self.agent_timesteps_since_eval += len(randomized_trajectory[i])

        # Calculate discriminator based reward, pass it back to SVPG policy
        if self.svpg_timesteps >= self.initial_svpg_steps: 
            if self.train_svpg:
                # self.svpg.train(rewards)
                #### !!!update sampling strategy############
                randomized_returns = np.array(randomized_returns) # (5,10) (self.svpg.svpg_rollout_length,self.nagents)
                if self.args.use_dist:
                    randomized_returns = - np.array(randomized_dists) # (5,10) (self.svpg.svpg_rollout_length,self.nagents)
                randomized_returns_flatten = torch.FloatTensor(randomized_returns.reshape(self.svpg.svpg_rollout_length*self.nagents)).to(device)
                simulation_instances_flatten = torch.FloatTensor(simulation_instances.reshape(self.svpg.svpg_rollout_length*self.nagents, self.svpg.nparams)).to(device)
                acquisition_score, acquisition_mean, acquisition_std = self.sampler.get_acquisition_score(simulation_instances_flatten) # ntask,1

                randomized_returns_flatten = - randomized_returns_flatten
                if not self.args.no_batch_norm:
                    y = (randomized_returns_flatten - randomized_returns_flatten.mean()) / (randomized_returns_flatten.std() + 1e-8)
                else:
                    y = randomized_returns_flatten
                sampler_loss, recon_loss, kl_loss = 0, 0, 0
                for _ in range(self.args.sampler_train_times):
                    sampler_loss, recon_loss, kl_loss = self.sampler.train(simulation_instances_flatten, y)
                sampler_loss += sampler_loss
                recon_loss += recon_loss
                kl_loss += kl_loss
                ############################################

            for dimension in range(self.nparams):
                self.sampled_regions[dimension] = np.concatenate([
                    self.sampled_regions[dimension], simulation_instances[:, :, dimension].flatten()
                ])           

        solved_reference = info = None
        if self.agent_timesteps_since_eval >= self.agent_eval_frequency:
            self.agent_timesteps_since_eval %= self.agent_eval_frequency
            logger.info("Evaluating for {} episodes afer timesteps: {} (SVPG), {} (Agent)".format(
                self.randomized_eval_episodes * self.nagents, self.svpg_timesteps, self.agent_timesteps))

            agent_reference_eval_rewards = []
            agent_hard_eval_rewards = []
            agent_randomized_eval_rewards = []
            avg_agent_randomized_eval_rewards = []

            final_dist_ref = []
            final_dist_hard = []
            final_dist_rand = []
            avg_final_dist_rand = []
            eval_acquisition_scores = []

            rewards_ref, dist_ref = evaluate_policy(nagents=self.nagents,
                                                    env=self.reference_env,
                                                    agent_policy=agent_policy,
                                                    replay_buffer=None,
                                                    eval_episodes=5,
                                                    max_steps=self.max_env_timesteps,
                                                    return_rewards=True,
                                                    add_noise=False,
                                                    log_distances=self.log_distances)
            rewards_hard, dist_hard = evaluate_policy(nagents=self.nagents,
                                                    env=self.hard_env,
                                                    agent_policy=agent_policy,
                                                    replay_buffer=None,
                                                    eval_episodes=5,
                                                    max_steps=self.max_env_timesteps,
                                                    return_rewards=True,
                                                    add_noise=False,
                                                    log_distances=self.log_distances)
            agent_reference_eval_rewards += list(rewards_ref)
            agent_hard_eval_rewards += list(rewards_hard)
            final_dist_ref += [dist_ref]
            final_dist_hard += [dist_hard]

            for _ in range(self.randomized_eval_episodes):
                full_random_settings = np.random.uniform(0.0, 1.0, size=(self.nagents, self.nparams))
                self.randomized_eval_env.randomize(randomized_values=full_random_settings)
                full_random_settings_tensor = torch.FloatTensor(full_random_settings).to(device)
                eval_acquisition_score, _, _ = self.sampler.get_acquisition_score(full_random_settings_tensor)

                rewards_rand, dist_rand = evaluate_policy(nagents=self.nagents,
                                                          env=self.randomized_eval_env,
                                                          agent_policy=agent_policy,
                                                          replay_buffer=None,
                                                          eval_episodes=5,
                                                          max_steps=self.max_env_timesteps,
                                                          return_rewards=True,
                                                          add_noise=False,
                                                          log_distances=self.log_distances)
                avg_rewards_rand = np.mean(rewards_rand.reshape(5,-1),axis=0)
                avg_dist_rand = np.mean(dist_rand.reshape(5,-1),axis=0)

                agent_randomized_eval_rewards += list(rewards_rand)
                final_dist_rand += [dist_rand]
                avg_agent_randomized_eval_rewards += list(avg_rewards_rand)
                avg_final_dist_rand += [avg_dist_rand]
                eval_acquisition_scores += list(eval_acquisition_score.squeeze().cpu().detach().numpy())

            evaluation_criteria_reference = agent_reference_eval_rewards
            evaluation_criteria_randomized = agent_randomized_eval_rewards

            if self.log_distances:
                evaluation_criteria_reference = final_dist_ref
                evaluation_criteria_randomized = final_dist_rand

            solved_reference = check_solved(self.reference_env_id, evaluation_criteria_reference)
            solved_randomized = check_solved(self.randomized_eval_env_id, evaluation_criteria_randomized)

            info = {
                'solved': str(solved_reference),
                'solved_randomized': str(solved_randomized),
                'svpg_steps': self.svpg_timesteps,
                'agent_timesteps': self.agent_timesteps,
                'final_dist_ref_mean': np.mean(final_dist_ref),
                'final_dist_ref_std': np.std(final_dist_ref),
                'final_dist_ref_median': np.median(final_dist_ref),
                'final_dist_hard_mean': np.mean(final_dist_hard),
                'final_dist_hard_std': np.std(final_dist_hard),
                'final_dist_hard_median': np.median(final_dist_hard),
                'final_dist_rand_mean': np.mean(final_dist_rand),
                'final_dist_rand_std': np.std(final_dist_rand),
                'final_dist_rand_median': np.median(final_dist_rand),
                'agent_reference_eval_rewards_mean': np.mean(agent_reference_eval_rewards),
                'agent_reference_eval_rewards_std': np.std(agent_reference_eval_rewards),
                'agent_reference_eval_rewards_median': np.median(agent_reference_eval_rewards),
                'agent_reference_eval_rewards_min': np.min(agent_reference_eval_rewards),
                'agent_reference_eval_rewards_max': np.max(agent_reference_eval_rewards),
                'agent_hard_eval_rewards_median': np.median(agent_hard_eval_rewards),
                'agent_hard_eval_rewards_mean': np.mean(agent_hard_eval_rewards),
                'agent_hard_eval_rewards_std': np.std(agent_hard_eval_rewards),
                'agent_randomized_eval_rewards_mean': np.mean(agent_randomized_eval_rewards),
                'agent_randomized_eval_rewards_std': np.std(agent_randomized_eval_rewards),
                'agent_randomized_eval_rewards_median': np.median(agent_randomized_eval_rewards),
                'agent_randomized_eval_rewards_min': np.min(agent_randomized_eval_rewards),
                'agent_randomized_eval_rewards_max': np.max(agent_randomized_eval_rewards),
                'acquisition_mean':acquisition_mean.mean().item(),
                'acquisition_std':acquisition_std.mean().item(),
                'return_corr':np.corrcoef(acquisition_score.squeeze().cpu().detach().numpy(), randomized_returns_flatten.squeeze().cpu().detach().numpy())[0,1],  
                'sampler_loss': sampler_loss, 
                'recon_loss': recon_loss,
                'kl_loss':kl_loss,
            }
            cvar50_avg_agent_randomized_eval_rewards = np.sort(avg_agent_randomized_eval_rewards)[:int(len(avg_agent_randomized_eval_rewards) * 0.5)]
            cvar30_avg_agent_randomized_eval_rewards = np.sort(avg_agent_randomized_eval_rewards)[:int(len(avg_agent_randomized_eval_rewards) * 0.3)]
            cvar10_avg_agent_randomized_eval_rewards = np.sort(avg_agent_randomized_eval_rewards)[:int(len(avg_agent_randomized_eval_rewards) * 0.1)]
            cvar5_avg_agent_randomized_eval_rewards = np.sort(avg_agent_randomized_eval_rewards)[:int(len(avg_agent_randomized_eval_rewards) * 0.05)]
            wandb.log({'eval_hard_rewards': np.mean(agent_hard_eval_rewards),
                'eval/reference_rewards': np.mean(agent_reference_eval_rewards),
                'eval/unif_rewards': np.mean(agent_randomized_eval_rewards),
                'eval/cvar50_rewards': np.mean(cvar50_avg_agent_randomized_eval_rewards),
                'eval/cvar30_rewards': np.mean(cvar30_avg_agent_randomized_eval_rewards),
                'eval/cvar10_rewards': np.mean(cvar10_avg_agent_randomized_eval_rewards),
                'eval/cvar5_rewards': np.mean(cvar5_avg_agent_randomized_eval_rewards),
                'train/svpg_timesteps': self.svpg_timesteps,
                'train/sampled_mean': simulation_instances_flatten.mean().item(),
                'train/sampled_0.25_ratio': (1.0*(simulation_instances_flatten<=0.25)).mean().item(),
                'train/return_corr': np.corrcoef(acquisition_score.squeeze().cpu().detach().numpy(), randomized_returns_flatten.squeeze().cpu().detach().numpy())[0,1],
                'eval/sampler_return_corr': np.corrcoef(np.array(eval_acquisition_scores), -np.array(avg_agent_randomized_eval_rewards))[0,1],
                'train_sampler/sampler_loss': sampler_loss,
                'train_sampler/recon_loss': recon_loss,
                'train_sampler/kl_loss': kl_loss,
                'step': self.agent_timesteps,})

        self.svpg_timesteps += 1
        return solved_reference, info

    def rollout_agent(self, agent_policy, reference=True, eval_episodes=None, cvar=None, gdroweight=None):
        """Rolls out agent_policy in the specified environment
        """
        if cvar:
            trajectory,avg_returns,avg_dists,env_steps = cvar_evaluate_policy(nagents=self.nagents,
                                env=self.randomized_env,
                                agent_policy=agent_policy,
                                replay_buffer=self.replay_buffer,
                                eval_episodes=self.episodes_per_instance,
                                max_steps=self.max_env_timesteps,
                                freeze_agent=self.freeze_agent,
                                add_noise=True,
                                log_distances=self.log_distances,
                                cvar=cvar)
            return trajectory,avg_returns,avg_dists
        elif gdroweight:
            trajectory,avg_returns,avg_dists = gdro_evaluate_policy(nagents=self.nagents,
                                env=self.randomized_env,
                                agent_policy=agent_policy,
                                replay_buffer=self.replay_buffer,
                                eval_episodes=self.episodes_per_instance,
                                max_steps=self.max_env_timesteps,
                                freeze_agent=self.freeze_agent,
                                add_noise=True,
                                log_distances=self.log_distances,
                                gdroweight=gdroweight)
            return trajectory,avg_returns,avg_dists
        else:
            if reference:
                if eval_episodes is None:
                    eval_episodes = self.episodes_per_instance
                trajectory,avg_returns,avg_dists = evaluate_policy(nagents=self.nagents,
                                            env=self.reference_env,
                                            agent_policy=agent_policy,
                                            replay_buffer=None,
                                            eval_episodes=eval_episodes,
                                            max_steps=self.max_env_timesteps,
                                            freeze_agent=True,
                                            add_noise=False,
                                            log_distances=self.log_distances)
            else:
                trajectory,avg_returns,avg_dists = evaluate_policy(nagents=self.nagents,
                                            env=self.randomized_env,
                                            agent_policy=agent_policy,
                                            replay_buffer=self.replay_buffer,
                                            eval_episodes=self.episodes_per_instance,
                                            max_steps=self.max_env_timesteps,
                                            freeze_agent=self.freeze_agent,
                                            add_noise=True,
                                            log_distances=self.log_distances)

            return trajectory,avg_returns,avg_dists

    def sample_trajectories(self, batch_size):
        indices = np.random.randint(0, len(self.extracted_trajectories['states']), batch_size)

        states = self.extracted_trajectories['states']
        actions = self.extracted_trajectories['actions']
        next_states = self.extracted_trajectories['next_states']

        trajectories = []
        for i in indices:
            trajectories.append(np.concatenate(
                [
                    np.array(states[i]),
                    np.array(actions[i]),
                    np.array(next_states[i])
                ], axis=-1))
        return trajectories

    def evaluate_in_full_range(self, agent_policy):
        assert self.reference_env_id.find('Lunar') != -1
        self.full_eval_env = make_vec_envs('LunarLanderRandomizedFull-v0', self.seed, self.nagents)
        rewards_grids = []
        dist_grids = []
        for tau in np.linspace(0.0, 1.0, 23):
            full_settings = np.ones((self.nagents, self.nparams)) * tau
            self.full_eval_env.randomize(randomized_values=full_settings)

            rewards_grid, dist_grid = evaluate_policy(nagents=self.nagents,
                                                        env=self.full_eval_env,
                                                        agent_policy=agent_policy,
                                                        replay_buffer=None,
                                                        eval_episodes=5,
                                                        max_steps=self.max_env_timesteps,
                                                        return_rewards=True,
                                                        add_noise=False,
                                                        log_distances=self.log_distances)

            tau_scaled = self.full_eval_env.rescale(0, tau)
            wandb.log({'test_full_range/rewards': np.mean(rewards_grid),
                        # 'test_full_range/dist': np.mean(dist_grid),
                        'tau_scaled': int(round(tau_scaled))})
            rewards_grids.append(np.mean(rewards_grid))
        full_tau_scaled = self.full_eval_env.rescale(0, np.linspace(0.0, 1.0, 23))
        return np.array(rewards_grids), full_tau_scaled
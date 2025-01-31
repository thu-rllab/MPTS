import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.models.actor_critic import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG(object):
    def __init__(self, state_dim, action_dim, agent_name='baseline', max_action=1.):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=3e-4)

        self.max_action = max_action

        self.agent_name = agent_name

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy()

    def train(self, replay_buffer, iterations, training_iters=1, batch_size=256, discount=0.99, tau=0.005, gdroweight=None, cvar=None):
        for it in range(iterations):
            # Sample replay buffer 
            if gdroweight is not None:
                x, y, u, r, d, ret = replay_buffer.sample(batch_size, gdroweight)
                returns = torch.FloatTensor(ret).to(device)
            else:
                x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(action) * 0.2
                ).clamp(-0.5, 0.5)
                next_action = (
                    self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            if (training_iters if cvar is None and gdroweight is None else (it+1)) % 2 == 0:
                if gdroweight is not None:
                    adv_probs = torch.ones(1).to(device)
                    adv_probs = adv_probs * torch.exp(gdroweight * returns)
                    adv_probs = adv_probs/torch.sum(adv_probs)
                    actor_loss = (-self.critic.Q1(state, self.actor(state)) * adv_probs).sum()
                else:
                    # Compute actor loss
                    actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location=device))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename), map_location=device))

    # To ensure backwards compatibility D:
    def load_model(self):
        cur_dir = os.getcwd()
        actor_path = 'common/agents/ddpg/saved_model/{}_{}.pth'.format(self.agent_name, 'actor')
        critic_path = 'common/agents/ddpg/saved_model/{}_{}.pth'.format(self.agent_name, 'critic')

        self.actor.load_state_dict(torch.load(os.path.join(cur_dir, actor_path), map_location=device))
        self.critic.load_state_dict(torch.load(os.path.join(cur_dir, critic_path), map_location=device))

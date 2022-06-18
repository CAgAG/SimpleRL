# @Date    : 2022/3/26
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import os

import torch
from torch import optim
import torch.nn.functional as F

from model import Actor, Critic
from utils import device


class DDPG:
    def __init__(self, obs_dim, act_dim, max_action,
                 gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(obs_dim, act_dim, max_action).to(device)
        self.actor_target = Actor(obs_dim, act_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = Critic(obs_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

    def predict(self, obs):
        obs = obs.to(device)
        return self.actor(obs)

    def learn(self, obs, act, reward, done, next_obs):
        # Compute the target Q value
        target_Q = self.critic_target(next_obs, self.actor_target(next_obs))
        target_Q = reward + (done * self.gamma * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(obs, act)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, file_dir: str, episode):
        actor_path = os.path.join(file_dir, f'actor-{episode}.ckpt')
        critic_path = os.path.join(file_dir, f'critic-{episode}.ckpt')
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, file_dir: str, episode):
        actor_path = os.path.join(file_dir, f'actor-{episode}.ckpt')
        critic_path = os.path.join(file_dir, f'critic-{episode}.ckpt')
        self.actor.load_state_dict(torch.load(actor_path))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(torch.load(critic_path))
        self.critic_target.load_state_dict(self.critic.state_dict())

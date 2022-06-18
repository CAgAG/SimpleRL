# @Date    : 2022/3/27
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import os

import torch
from torch import optim
import torch.nn.functional as F

from model import Actor, Critic
from utils import device


class TD3:
    def __init__(self, obs_dim, act_dim, max_action, ):
        self.actor = Actor(obs_dim, act_dim, max_action).to(device)
        self.critic_1 = Critic(obs_dim, act_dim).to(device)
        self.critic_2 = Critic(obs_dim, act_dim).to(device)

        self.actor_target = Actor(obs_dim, act_dim, max_action).to(device)
        self.critic_1_target = Critic(obs_dim, act_dim).to(device)
        self.critic_2_target = Critic(obs_dim, act_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters())

        self.max_action = max_action
        self.tau = 0.005
        self.gamma = 0.99
        self.noise_clip = 0.5
        self.policy_noise = 0.2
        self.policy_delay = 2

    def predict(self, obs):
        obs = obs.to(device)
        return self.actor(obs)

    def learn(self, obs, act, reward, done, next_obs, delay_actor_i):
        # Select next action according to target policy:
        noise = torch.ones_like(act).data.normal_(0, self.policy_noise).to(device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_action = self.actor_target(next_obs) + noise
        next_action = next_action.clamp(-self.max_action, self.max_action)

        # Compute target Q-value:
        target_Q1 = self.critic_1_target(next_obs, next_action)
        target_Q2 = self.critic_2_target(next_obs, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        # Optimize Critic 1:
        current_Q1 = self.critic_1(obs, act)
        loss_Q1 = F.mse_loss(current_Q1, target_Q)
        self.critic_1_optimizer.zero_grad()
        loss_Q1.backward()
        self.critic_1_optimizer.step()

        # Optimize Critic 2:
        current_Q2 = self.critic_2(obs, act)
        loss_Q2 = F.mse_loss(current_Q2, target_Q)
        self.critic_2_optimizer.zero_grad()
        loss_Q2.backward()
        self.critic_2_optimizer.step()

        # Delayed policy updates:
        if delay_actor_i % self.policy_delay == 0:
            # Compute actor loss:
            actor_loss = -self.critic_1(obs, self.actor(obs)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

    def save(self, file_dir: str, episode):
        actor_path = os.path.join(file_dir, f'actor-{episode}.ckpt')
        critic_1_path = os.path.join(file_dir, f'critic-1-{episode}.ckpt')
        critic_2_path = os.path.join(file_dir, f'critic-2-{episode}.ckpt')
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic_1.state_dict(), critic_1_path)
        torch.save(self.critic_2.state_dict(), critic_2_path)

    def load(self, file_dir: str, episode):
        actor_path = os.path.join(file_dir, f'actor-{episode}.ckpt')
        critic_1_path = os.path.join(file_dir, f'critic-1-{episode}.ckpt')
        critic_2_path = os.path.join(file_dir, f'critic-2-{episode}.ckpt')
        self.actor.load_state_dict(torch.load(actor_path))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1.load_state_dict(torch.load(critic_1_path))
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2.load_state_dict(torch.load(critic_2_path))
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

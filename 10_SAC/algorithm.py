# @Date    : 2022/3/28
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import os

import torch
from torch import nn
from torch import optim
from torch.distributions import Normal

from model import Actor, Critic, Q
from utils import device


class SAC:
    def __init__(self, obs_dim, act_dim, max_action):
        self.policy_net = Actor(obs_dim, max_action=max_action).to(device)
        self.value_net = Critic(obs_dim).to(device)
        self.Q_net = Q(obs_dim, act_dim).to(device)
        self.Target_value_net = Critic(obs_dim).to(device)

        learning_rate = 3e-4
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=learning_rate)

        self.value_criterion = nn.MSELoss()
        self.Q_criterion = nn.MSELoss()

        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.min_Val = torch.tensor(1e-7).float()
        self.gamma = 0.99
        self.tau = 0.005

    def predict(self, obs):
        obs = obs.to(device)
        mu, log_sigma = self.policy_net(obs)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        return dist

    def predict_act_log_prob(self, obs):
        dist = self.predict(obs)
        z = dist.sample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + self.min_Val)
        return action, log_prob

    def learn(self, obs, act, reward, done, next_obs):
        target_value = self.Target_value_net(next_obs)
        next_q_value = reward + (1 - done) * self.gamma * target_value

        excepted_value = self.value_net(obs)
        excepted_Q = self.Q_net(obs, act)

        sample_action, log_prob = self.predict_act_log_prob(obs)
        excepted_sample_Q = self.Q_net(obs, sample_action)
        next_value = excepted_sample_Q - log_prob

        # !!!Note that the actions are sampled according to the current policy,
        # instead of replay buffer. (From original paper)
        V_loss = self.value_criterion(excepted_value, next_value.detach())  # J_V
        V_loss = V_loss.mean()

        # Single Q_net this is different from original paper!!!
        Q_loss = self.Q_criterion(excepted_Q, next_q_value.detach())  # J_Q
        Q_loss = Q_loss.mean()

        log_policy_target = excepted_sample_Q - excepted_value

        pi_loss = log_prob * (log_prob - log_policy_target).detach()
        pi_loss = pi_loss.mean()

        # mini batch gradient descent
        self.value_optimizer.zero_grad()
        V_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()

        self.Q_optimizer.zero_grad()
        Q_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.Q_net.parameters(), 0.5)
        self.Q_optimizer.step()

        self.policy_optimizer.zero_grad()
        pi_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()

        # soft update
        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)

    def save(self, file_dir: str, episode):
        actor_path = os.path.join(file_dir, f'actor-{episode}.ckpt')
        value_path = os.path.join(file_dir, f'critic-{episode}.ckpt')
        q_path = os.path.join(file_dir, f'q-{episode}.ckpt')
        torch.save(self.policy_net.state_dict(), actor_path)
        torch.save(self.value_net.state_dict(), value_path)
        torch.save(self.Q_net.state_dict(), q_path)

    def load(self, file_dir: str, episode):
        actor_path = os.path.join(file_dir, f'actor-{episode}.ckpt')
        value_path = os.path.join(file_dir, f'critic-{episode}.ckpt')
        q_path = os.path.join(file_dir, f'q-{episode}.ckpt')
        self.policy_net.load_state_dict(torch.load(actor_path))
        self.value_net.load_state_dict(torch.load(value_path))
        self.Q_net.load_state_dict(torch.load(q_path))
        self.Target_value_net.load_state_dict(torch.load(value_path))

# @Date    : 2022/3/18
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import numpy as np
import torch
from torch import nn
from collections import deque
import torch.nn.functional as F

GAMMA = 0.99
eps = np.finfo(np.float32).eps.item()


class AC:
    def __init__(self, model: nn.Module, lr=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(
            lr=lr, params=self.model.parameters()
        )

    def predict(self, obs):
        act_prob, critic_prob = self.model(obs)
        return act_prob, critic_prob

    def learn(self, sample_data, reward_list):
        R = 0
        save_actions = sample_data
        policy_loss = []
        value_loss = []
        rewards = deque()

        for r in reward_list[::-1]:
            R = r + GAMMA * R
            rewards.appendleft(R)

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        for (log_prob, critic_prob), r in zip(save_actions, rewards):
            reward = r - critic_prob.item()
            policy_loss.append(-log_prob * reward)
            value_loss.append(F.smooth_l1_loss(critic_prob, torch.tensor([r])))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        self.optimizer.step()
        return loss

    def save(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path))

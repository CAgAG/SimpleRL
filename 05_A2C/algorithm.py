# @Date    : 2022/3/18
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class AC:
    def __init__(self, model: nn.Module, lr=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(
            lr=lr, params=self.model.parameters()
        )

    def predict(self, obs):
        dist, critic_prob = self.model(obs)
        return dist, critic_prob

    def learn(self, log_probs, advantage, entropy):
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path))

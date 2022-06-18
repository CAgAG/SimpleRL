# @Date    : 2022/3/26
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import torch
from torch import nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(obs_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, act_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(obs_dim + act_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

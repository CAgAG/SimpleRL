# @Date    : 2022/2/25
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super(Model, self).__init__()
        h1_dim = act_dim * 10
        self.fc_1 = nn.Linear(obs_dim, h1_dim)
        self.fc_2 = nn.Linear(h1_dim, act_dim)

    def forward(self, obs):
        out = torch.tanh(self.fc_1(obs))
        prob = F.softmax(self.fc_2(out), dim=-1)
        return prob
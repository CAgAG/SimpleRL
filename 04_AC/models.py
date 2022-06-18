# @Date    : 2022/3/18
# @Author  : CAgAG
# @Version : 1.0
# @Function:

from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 32)

        self.action_head = nn.Linear(32, act_dim)
        self.value_head = nn.Linear(32, 1)  # Scalar Value

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)

        return F.softmax(action_score, dim=-1), state_value


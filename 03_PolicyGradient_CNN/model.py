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
        h1_dim = act_dim * act_dim

        self.conv_1 = nn.Conv2d(obs_dim, h1_dim, kernel_size=(8, 8), stride=(4, 4))
        self.conv_2 = nn.Conv2d(h1_dim, h1_dim, kernel_size=(4, 4), stride=(2, 2))
        self.conv_3 = nn.Conv2d(h1_dim, h1_dim, kernel_size=(3, 3), stride=(1, 1))
        self.fc_1 = nn.Linear(6 * 6 * 36, act_dim)

    def forward(self, obs):
        out = F.relu(self.conv_1(obs))
        out = F.relu(self.conv_2(out))
        out = F.relu(self.conv_3(out))

        out = torch.flatten(out, 1)
        prob = F.softmax(self.fc_1(out), dim=-1)
        return prob


if __name__ == '__main__':
    data = torch.rand((1, 1, 80, 80))

    model = Model(obs_dim=1, act_dim=4)
    out = model(data)

    print(out.shape)
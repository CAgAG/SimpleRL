import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        h1_dim = 128
        h2_dim = 128

        self.fc_1 = nn.Linear(obs_dim, h1_dim)
        self.fc_2 = nn.Linear(h1_dim, h2_dim)
        self.fc_3 = nn.Linear(h2_dim, act_dim)

    def forward(self, obs):
        h1 = F.relu(self.fc_1(obs))
        h2 = F.relu(self.fc_2(h1))
        Q = self.fc_3(h2)
        return Q
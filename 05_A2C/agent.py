# @Date    : 2022/3/18
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import numpy as np

import torch
from torch.distributions import Categorical

eps = np.finfo(np.float32).eps.item()


class Agent:
    def __init__(self, alg, device=torch.device('cpu')):
        self.alg = alg
        self.log_probs = []
        self.q_values = []
        self.entropy = 0

        self.device = device

    def predict_act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        dist, _ = self.alg.predict(obs)
        return dist.sample().cpu().numpy()

    def predict_obs(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, prob = self.alg.predict(obs)
            value = prob.cpu().numpy()[0]
            return float(value)

    def sample(self, obs):
        state = torch.from_numpy(obs).float().to(self.device)
        dist, state_value = self.alg.predict(state)

        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(1)

        self.log_probs.append(log_prob)
        self.q_values.append(state_value)
        self.entropy += dist.entropy().mean()

        return action.cpu().numpy()

    def learn(self, returns):
        log_probs = torch.cat(self.log_probs).to(self.device)
        q_values = torch.cat(self.q_values).to(self.device)
        returns = torch.cat(list(returns)).detach().to(self.device)

        advantage = returns - q_values

        loss = self.alg.learn(log_probs, advantage, self.entropy)

        self.log_probs.clear()
        self.q_values.clear()
        self.entropy = 0
        return float(loss.mean().cpu().detach().numpy())

    def save(self, file_path: str):
        self.alg.save(file_path)

    def load(self, file_path: str):
        self.alg.load(file_path)

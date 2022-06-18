# @Date    : 2022/3/18
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import numpy as np

import torch
from torch.distributions import Categorical


class Agent:
    def __init__(self, alg):
        self.alg = alg
        self.sample_data = []

    def predict_act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        prob, _ = self.alg.predict(obs)
        # 根据动作概率选择概率最高的动作
        act = prob.argmax(keepdim=True).numpy()[0]
        return act

    def predict_obs(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        _, prob = self.alg.predict(obs)
        value = prob.numpy()[0]
        return value

    def sample(self, obs):
        state = torch.from_numpy(obs).float()
        probs, state_value = self.alg.predict(state)

        m = Categorical(probs)
        action = m.sample()
        self.sample_data.append((m.log_prob(action), state_value))

        return action.item()

    def learn(self, reward_list):
        loss = self.alg.learn(self.sample_data, reward_list)
        self.sample_data.clear()
        return float(loss.data.numpy())

    def save(self, file_path: str):
        self.alg.save(file_path)

    def load(self, file_path: str):
        self.alg.load(file_path)

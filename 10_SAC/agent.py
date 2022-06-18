# @Date    : 2022/3/28
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import numpy as np
import torch

from algorithm import SAC
from utils import device


class Agent:
    def __init__(self, alg: SAC):
        self.alg = alg

        self.gradient_steps = 1
        self.batch_size = 128

    def predict_act(self, obs):
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            dist = self.alg.predict(obs)
            z = dist.sample()
            action = torch.tanh(z).detach().cpu().numpy()
            return action.item()  # return a scalar, float32

    def learn(self, buffer):
        for _ in range(self.gradient_steps):
            obs, act, next_obs, reward, done = buffer.sample(self.batch_size)

            batch_obs = torch.tensor(obs, dtype=torch.float32).to(device)
            batch_act = torch.tensor(act, dtype=torch.float32).to(device)
            batch_reward = torch.tensor(reward, dtype=torch.float32).to(device)
            batch_next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            batch_done = torch.tensor(done, dtype=torch.float32).to(device)

            self.alg.learn(batch_obs, batch_act, batch_reward, batch_done, batch_next_obs)

    def save(self, file_dir: str, episode):
        self.alg.save(file_dir, episode)

    def load(self, file_dir: str, episode):
        self.alg.load(file_dir, episode)

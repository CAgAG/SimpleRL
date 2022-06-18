# @Date    : 2022/3/26
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import torch

from utils import device


class Agent:
    def __init__(self, alg, update_iter=200, batch_size=100):
        self.alg = alg
        self.update_iter = update_iter
        self.batch_size = batch_size

    def predict(self, obs):
        obs = torch.FloatTensor(obs.reshape(1, -1))
        with torch.no_grad():
            act = self.alg.predict(obs)
        return act.cpu().data.numpy().flatten()

    def learn(self, buffer):
        for it in range(self.update_iter):
            # Sample replay buffer
            x, y, u, r, d = buffer.sample(self.batch_size)
            obs = torch.FloatTensor(x).to(device)
            act = torch.FloatTensor(u).to(device)
            next_obs = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            self.alg.learn(obs, act, reward, done, next_obs)

    def save(self, file_dir: str, episode):
        self.alg.save(file_dir, episode)

    def load(self, file_dir: str, episode):
        self.alg.load(file_dir, episode)

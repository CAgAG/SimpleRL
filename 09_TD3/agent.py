# @Date    : 2022/3/27
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import torch

from algorithm import TD3
from utils import device


class Agent:
    def __init__(self, alg: TD3, num_iter=10, batch_size=100):
        self.alg = alg
        self.batch_size = batch_size
        self.num_iter = num_iter

    def predict_act(self, obs):
        obs = torch.tensor(obs.reshape(1, -1)).float()
        with torch.no_grad():
            act = self.alg.predict(obs)
        return act.cpu().data.numpy().flatten()

    def learn(self, buffer):
        for i in range(self.num_iter):
            x, y, u, r, d = buffer.sample(self.batch_size)
            obs = torch.FloatTensor(x).to(device)
            act = torch.FloatTensor(u).to(device)
            next_obs = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            self.alg.learn(obs, act, reward, done, next_obs, delay_actor_i=i)

    def save(self, file_dir: str, episode):
        self.alg.save(file_dir, episode)

    def load(self, file_dir: str, episode):
        self.alg.load(file_dir, episode)

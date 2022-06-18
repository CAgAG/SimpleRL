# @Date    : 2022/2/25
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import torch
import numpy as np


class Agent:
    def __init__(self, alg, mode='linear'):
        super(Agent, self).__init__()
        self.alg = alg
        self.mode = mode

    def predict(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        prob = self.alg.predict(obs)
        # 根据动作概率选择概率最高的动作
        act = prob.argmax(keepdim=True).numpy()[0]
        return act

    def sample(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        prob = self.alg.predict(obs).detach().numpy()

        # 根据动作概率选取动作
        act = np.random.choice(len(prob), 1, p=prob)[0]
        return act

    def learn(self, obs, act, reward):
        if self.mode == 'linear':
            act = np.expand_dims(act, axis=-1)
            reward = np.expand_dims(reward, axis=-1)

        obs = torch.tensor(obs, dtype=torch.float32)
        act = torch.tensor(act, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)

        loss = self.alg.learn(obs, act, reward)
        return float(loss.data.numpy())

    def save(self, file_path: str):
        self.alg.save(file_path)

    def load(self, file_path: str):
        self.alg.load(file_path)

import torch
import numpy as np


class Agent:
    def __init__(self, algorithm,
                 act_dim: int, e_greed: float = 0.1, e_greed_dec: float = 0.01):
        super(Agent, self).__init__()
        self.alg = algorithm
        self.act_dim = act_dim
        self.global_steps = 0

        # 每隔 200 个training steps再把 model 的参数复制到 target_model 中 (fix target model)
        self.update_target_steps = 200
        self.e_greed = e_greed
        self.e_greed_dec = e_greed_dec

    def predict(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        pred_q = self.alg.predict(obs)
        act = pred_q.argmax(keepdim=True).numpy()[0]
        return act

    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作
        """
        sample = np.random.random()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_dec)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def learn(self, obs, act, reward, next_obs, done):
        # 准备训练数据
        # 更新 target model
        if self.global_steps % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_steps += 1

        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        done = np.expand_dims(done, axis=-1)

        obs = torch.tensor(obs, dtype=torch.float32)
        act = torch.tensor(act, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # 开始训练
        loss = self.alg.learn(obs, act, reward, next_obs, done)
        return float(loss.data.numpy())

    def save(self, file_path: str):
        self.alg.save(file_path)

    def load(self, file_path: str):
        self.alg.load(file_path)

# @Date    : 2022/2/25
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import torch
import torch.nn.functional as F


class PolicyGradient:
    def __init__(self, model, lr: float, device):
        super(PolicyGradient, self).__init__()
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            lr=lr, params=self.model.parameters()
        )

    def predict(self, obs):
        obs = obs.to(self.device)
        prob = self.model(obs)
        return prob

    def learn(self, obs, action, reward):
        # 获取输出动作概率
        prob = self.predict(obs)
        action = action.to(self.device)
        reward = reward.to(self.device)

        # log_prob = Categorical(prob).log_prob(action) 交叉熵
        # loss = paddle.mean(-1 * log_prob * reward)
        action_onehot = torch.squeeze(
            F.one_hot(action, num_classes=prob.shape[-1]),
            dim=1
        )
        log_prob = torch.sum(torch.log(prob + 0.001) * action_onehot, dim=1)  # 加上 0.001 防止出现 log(0)=-inf 的情况
        if reward.shape != log_prob.shape:
            reward = torch.squeeze(reward, dim=1)
        loss = torch.mean(-1 * reward * log_prob)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path))
        self.model = self.model.to(self.device)

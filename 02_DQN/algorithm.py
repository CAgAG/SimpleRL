import torch
import torch.nn.functional as F


class DQN:
    def __init__(self, model, gamma: float, lr: float):
        super().__init__()
        self.model = model
        self.target_model = model

        self.gamma = gamma
        self.lr = lr

        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(
            lr=lr, params=self.model.parameters()
        )

    def predict(self, obs):
        return self.model(obs)

    def learn(self, obs, action, reward, next_obs, done):
        # 核心训练代码
        pred_q = self.predict(obs)

        action_dim = pred_q.shape[-1]
        action = torch.squeeze(action, dim=-1)

        # 转为 one hot
        action_onehot = F.one_hot(
            action, num_classes=action_dim
        )
        # 按位相乘
        # 比如：pred_q = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        pred_value = pred_q * action_onehot
        #  ==> pred_value = [[3.9]]
        pred_value = torch.sum(pred_value, dim=1, keepdim=True)

        # 固定 target model 防止更新
        with torch.no_grad():
            max_v = self.target_model(next_obs).max(1, keepdim=True).values
            # 1 - done: 是实现公式中，如果循环结束直接输出 reward
            target = reward + (1 - done) * self.gamma * max_v
        loss = self.mse_loss(pred_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path))
        self.target_model.load_state_dict(torch.load(file_path))

    def sync_target(self):
        # 把 self.model 的模型参数值同步到 self.target_model
        self.target_model.load_state_dict(self.model.state_dict())
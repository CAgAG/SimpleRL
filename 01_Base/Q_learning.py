# value base

import gym
import numpy as np
from Sarsa import SarsaAgent, test_episode


class QLearningAgent(SarsaAgent):
    def learn(self, obs, action, reward, next_obs, done):
        """ off-policy: 此处不需要 next_action """
        predict_q = self.Q[obs, action]  # Q_t
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.Q[next_obs, :])  # Q_t+1, 直接选取价值最高的 action
        self.Q[obs, action] += self.lr * (target_q - predict_q)


def run_episode(env, agent, render=False):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）

    while True:
        action = agent.sample(obs)  # 根据算法选择一个动作
        next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互
        # 训练 Q-learning算法
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1  # 计算step数
        if render:
            env.render()  # 渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps


if __name__ == '__main__':
    is_train = False
    # 使用gym创建悬崖环境
    env = gym.make("CliffWalking-v0")
    # 0 up, 1 right, 2 down, 3 left
    print(env.action_space)
    """
    observation是一个int，表示的是当前在第几格(位置)
    """

    agent = QLearningAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        lr=0.1,
        gamma=0.9,
        e_greed=0.1
    )

    if is_train:
        # train 500 episodes
        for episode in range(500):
            reward, step = run_episode(env, agent, False)
            print(f'Episode: [{episode:0>3d}|500], step: {step:0>3d}, reward: {reward}')
        agent.save(npy_file='./models/q_table2.npy')
    else:
        agent.load(npy_file='./models/q_table2.npy')
        reward = test_episode(env, agent)
        print(f'test reward: {reward}')

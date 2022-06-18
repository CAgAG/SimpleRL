# value base

import gym
import numpy as np


class SarsaAgent:
    def __init__(self, obs_n, act_n, lr=.01, gamma=.9, e_greed=.1):
        """
        :param obs_n: state/observation value  dimension
        :param act_n: action dimension
        :param lr: learning rate
        :param gamma: γ
        :param e_greed: 随机选动作概率
        """
        self.act_n = act_n
        self.lr = lr
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros((obs_n, act_n))

    def predict(self, obs):
        q_list = self.Q[obs, :]
        max_q = np.max(q_list)
        action_list = np.where(q_list == max_q)[0]  # max_q 可能对应多个action
        action = np.random.choice(action_list)
        return action

    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 有一定概率随机选择动作输出
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)
        return action

    def learn(self, obs, action, reward, next_obs, next_action, done):
        """ on-policy
        :param obs: 交互前的 state/observation
        :param action: 交互前的动作
        :param reward: 奖励
        :param next_obs: 交互后的 state/observation
        :param next_action: 交互后的动作
        :param done: 是否结束
        :return:
        """
        predict_q = self.Q[obs, action]  # Q_t
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * self.Q[next_obs, next_action]  # Sarsa, Q_t+1
        self.Q[obs, action] += self.lr * (target_q - predict_q)

    def save(self, npy_file='./models/q_table.npy'):
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def load(self, npy_file='./models/q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')


def run_episode(env, agent, render=False):
    total_steps = 0
    total_reward = 0

    obs = env.reset()
    action = agent.sample(obs)

    while True:
        next_obs, reward, done, _ = env.step(action)
        next_action = agent.sample(next_obs)

        # train
        agent.learn(obs, action, reward, next_obs, next_action, done)

        action = next_action
        obs = next_obs
        total_reward += reward
        total_steps += 1
        if render:
            print('================')
            env.render()
            print('================')
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()

    while True:
        action = agent.predict(obs)
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward

        obs = next_obs
        if done:
            break
    return total_reward


if __name__ == '__main__':
    is_train = False
    # 使用gym创建悬崖环境
    env = gym.make("CliffWalking-v0")
    # 0 up, 1 right, 2 down, 3 left
    print(env.action_space)
    """
    observation是一个int，表示的是当前在第几格(位置)
    """

    agent = SarsaAgent(
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
        agent.save()
    else:
        agent.load()
        reward = test_episode(env, agent)
        print(f'test reward: {reward}')

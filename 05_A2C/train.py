# @Date    : 2022/3/18
# @Author  : CAgAG
# @Version : 1.0
# @Function:
import random
from collections import deque

import numpy as np
import torch
import gym

from models import Model
from algorithm import AC
from agent import Agent
from multi_envs import SubprocVecEnv

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def compute_returns(next_value, rewards, dones, gamma=0.99):
    R = next_value
    returns = deque()
    for step in reversed(range(len(rewards))):
        reward = torch.FloatTensor(rewards[step]).unsqueeze(1)
        done = torch.FloatTensor(1 - dones[step]).unsqueeze(1)  # 取反
        R = reward + gamma * R * done
        returns.appendleft(R)
    return returns


def train_episode(agent, envs):
    obs_list, action_list, reward_list, next_obs_list, done_list = [], [], [], [], []
    num_done = [False for _ in range(len(envs))]
    obs = envs.reset()  # shape: (num worker, obs_dim)

    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        next_obs, reward, done, _ = envs.step(action)
        next_obs_list.append(next_obs)
        reward_list.append(reward)
        done_list.append(done)

        if any(done):
            for i_s in np.where(done):
                for i in i_s:
                    num_done[int(i)] = True
        if all(num_done):
            break
        obs = next_obs
    return obs_list, action_list, reward_list, next_obs_list, done_list


def eval_episode(agent, env):
    eval_reward = []

    for i in range(5):
        obs = env.reset()
        episode_reward = 0

        while True:
            action = agent.predict_act(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def show(agent, env):
    import cv2

    obs = env.reset()
    episode_reward = 0

    # 增加难度
    pre_action = random.randint(0, 1)
    for _ in range(8):
        obs, _, _, _ = env.step(pre_action)

    while True:
        action = agent.predict_act(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

        cv2.imshow('demo', env.render(mode='rgb_array'))
        cv2.waitKey(1)

        if done:
            cv2.destroyAllWindows()
            env.close()
            print(episode_reward)
            break


if __name__ == '__main__':
    SHOW = True

    save_path = './models/AC.ckpt'

    # CartPole-v1: expected reward > 475
    env = gym.make('CartPole-v1')  # for test

    # for train
    num_envs = 4
    env_name = "CartPole-v1"
    def make_env():
        def _thunk():
            env = gym.make(env_name)
            return env

        return _thunk
    envs = [make_env() for _ in range(num_envs)]
    envs = SubprocVecEnv(envs)  # 8 env

    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.n

    model = Model(obs_dim=obs_dim, act_dim=act_dim).to(device)
    AC_alg = AC(model=model, lr=0.001)
    agent = Agent(alg=AC_alg, device=device)

    if SHOW:
        agent.load(save_path)
        show(agent, env)
    else:
        for episode in range(2000):
            obs_list, action_list, reward_list, next_obs_list, done_list = train_episode(agent, envs)
            if episode % 10 == 0:
                print(f'Episode: {episode:0>3d}, Reward: {sum(reward_list)[0]}')

            next_value = agent.predict_obs(next_obs_list[-1])
            returns = compute_returns(next_value, reward_list, done_list)

            loss = agent.learn(returns)
            if (episode + 1) % 100 == 0:
                total_reward = eval_episode(agent, env)
                print(f'Test Reward: {total_reward}')
        agent.save(save_path)

# @Date    : 2022/3/17
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import gym
import numpy as np
import torch.cuda

from model import Model
from algorithm import PolicyGradient
from agent import Agent

LR = 1e-3
HISTORY_FRAME = 4

CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if CUDA else 'cpu')


def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into (3x80x80)"""
    image = image[35:195]  # 裁剪
    image = image[::2, ::2, 0]  # 下采样，缩放2倍
    image[image == 144] = 0  # 擦除背景 (background type 1)
    image[image == 109] = 0  # 擦除背景 (background type 2)
    image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色
    image = np.expand_dims(image, axis=2)  # shape -> (80, 80, 1)
    image = np.transpose(image, (2, 0, 1))
    return image


def calc_reward_to_go(reward_list, gamma=0.9):
    # MC 方式
    # G_i = r_i + γ·G_i+1
    for i in range(len(reward_list) - 2, -1, -1):
        reward_list[i] += gamma * reward_list[i + 1]  # G_t
    return np.array(reward_list)


def train_episode(agent, env):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()

    while True:
        obs = preprocess(obs)
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done, _ = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


def eval_episode(agent, env, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = preprocess(obs)
            obs = np.expand_dims(obs, axis=0)
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def show(agent, env):
    import cv2

    obs = env.reset()
    episode_reward = 0
    while True:
        obs = preprocess(obs)
        obs = np.expand_dims(obs, axis=0)
        action = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

        frame = env.render(mode='rgb_array')
        cv2.imshow('Pong', frame)
        cv2.waitKey(1)

        if done:
            cv2.destroyAllWindows()
            env.close()
            print(done, episode_reward)
            break


if __name__ == '__main__':
    SHOW = True

    save_path = './models/policy_gradient.ckpt'

    env = gym.make('ALE/Pong-v5')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = Model(obs_dim=1, act_dim=act_dim)
    alg = PolicyGradient(model, lr=LR, device=device)
    agent = Agent(alg)

    if SHOW:
        agent.load(save_path)
        show(agent, env)
    else:
        for episode in range(3000):
            obs_list, action_list, reward_list = train_episode(agent, env)
            if episode % 10 == 0:
                print(f'Episode: {episode:0>3d}, Reward: {sum(reward_list)}')

            batch_obs = np.array(obs_list)
            batch_action = np.array(action_list)
            batch_reward = calc_reward_to_go(reward_list)

            agent.learn(batch_obs, batch_action, batch_reward)
            if (episode + 1) % 100 == 0:
                total_reward = eval_episode(agent, env, render=False)
                print(f'Test Reward: {total_reward}')
        agent.save(save_path)

# @Date    : 2022/2/25
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import gym
import numpy as np

from model import Model
from algorithm import PolicyGradient
from agent import Agent

LR = 1e-3


def calc_reward_to_go(reward_list, gamma=1.0):
    # MC 方式
    # G_i = r_i + γ·G_i+1
    for i in range(len(reward_list) - 2, -1, -1):
        reward_list[i] += gamma * reward_list[i + 1]  # G_t
    return np.array(reward_list)


def train_episode(agent, env):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
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
        action = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

        cv2.imshow('demo', env.render(mode='rgb_array'))
        cv2.waitKey(1)

        if done:
            cv2.destroyAllWindows()
            env.close()
            print(done, episode_reward)
            break


if __name__ == '__main__':
    SHOW = True

    save_path = './models/policy_gradient.ckpt'
    # CartPole-v1: expected reward > 475
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    alg = PolicyGradient(model, lr=LR)
    agent = Agent(alg)

    if SHOW:
        agent.load(save_path)
        show(agent, env)
    else:
        for episode in range(2000):
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

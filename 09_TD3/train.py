# @Date    : 2022/3/27
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import time
import random

import cv2
import torch
import gym
import numpy as np

from algorithm import TD3
from agent import Agent
from replay_buffer import Replay_buffer

has_continuous_action_space = True
# expect reward >= -120
# Pendulum-v1
env_name = "Pendulum-v1"
env = gym.make(env_name)
# state space dimension
obs_dim = env.observation_space.shape[0]
# action space dimension
if has_continuous_action_space:
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
else:
    act_dim = env.action_space.n
    max_action = float(env.action_space.high[0])

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
env.seed(seed)
np.random.seed(seed)

exploration_noise = 0.1
act_high = env.action_space.high
act_low = env.action_space.low

if __name__ == '__main__':
    SHOW = True

    buffer = Replay_buffer()
    alg = TD3(obs_dim, act_dim, max_action=max_action)
    agent = Agent(alg, num_iter=10, batch_size=100)  # num_iter * batch_size, for update actor params

    if not SHOW:
        total_step = 0
        for episode in range(1, 3000 + 1):
            total_reward = 0
            obs = env.reset()
            while True:
                action = agent.predict_act(obs)
                # add noise
                action = (action + np.random.normal(0, exploration_noise, size=act_dim)).clip(
                    act_low, act_high)

                next_state, reward, done, info = env.step(action)
                buffer.push((obs, next_state, action, reward, done))

                obs = next_state
                total_reward += reward
                total_step += 1
                if done:
                    break

            if total_step % 10 * 100 == 0:
                agent.learn(buffer)
                print("Episode: {}, Total Reward: {:0.2f}".format(episode, total_reward))

            if episode % 500 == 0:
                agent.save('./models', episode)
    else:
        agent.load('./models', episode=3000)
        total_reward = 0
        obs = env.reset()

        for _ in range(6):
            act = random.randint(-2, 2)
            act = np.array([act], dtype=np.float32)
            obs, _, _, _ = env.step(act)

        while True:
            action = agent.predict_act(obs)

            next_state, reward, done, info = env.step(action)
            time.sleep(0.05)
            cv2.imshow('demo', env.render(mode='rgb_array'))
            cv2.waitKey(1)

            obs = next_state
            total_reward += reward
            if done:
                cv2.destroyAllWindows()
                env.close()
                break
        print("Total Reward: {:0.2f}".format(total_reward))

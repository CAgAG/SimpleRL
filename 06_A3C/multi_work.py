# @Date    : 2022/3/21
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import gym
import torch.multiprocessing as mp

from model import Net
from utils import to_dtype, train_one_episode, record

MAX_EP = 3000
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.99


class Worker(mp.Process):
    def __init__(self, gnet, optimizer, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name  # work unique name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.optimizer = gnet, optimizer

        self.env = gym.make('CartPole-v1').unwrapped
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n
        self.wnet = Net(obs_dim, act_dim)  # local network

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            obs = self.env.reset()
            buffer_obs, buffer_act, buffer_reward = [], [], []
            total_reward = 0.

            while True:
                action = self.wnet.choose_action(to_dtype(obs[None, :]))
                next_obs, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                total_reward += reward
                buffer_act.append(action)
                buffer_obs.append(obs)
                buffer_reward.append(reward)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync global net
                    train_one_episode(self.optimizer, self.wnet, self.gnet, done, next_obs,
                                      buffer_obs, buffer_act, buffer_reward, GAMMA)
                    buffer_obs.clear()
                    buffer_act.clear()
                    buffer_reward.clear()

                    if done:  # done and print information
                        record(self.g_ep, total_reward, self.res_queue, self.name)
                        break
                obs = next_obs
                total_step += 1
        print(f'{self.name} over')
        self.res_queue.put(None)

# @Date    : 2022/3/21
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import os

import gym
import torch
from torch import optim
import torch.multiprocessing as mp

from model import Net
from multi_work import Worker

os.environ["OMP_NUM_THREADS"] = "1"

env = gym.make('CartPole-v1')  # for obs_dim and act_dim
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

if __name__ == "__main__":
    gnet = Net(obs_dim, act_dim)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    optimizer = optim.Adam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    # global_ep: worker train n episode
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, optimizer, global_ep, global_ep_r, res_queue, i) for i in range(4)]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    torch.save(gnet.state_dict(), './models/A3C.ckpt')

# @Date    : 2022/3/26
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import numpy as np


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''

    def __init__(self, capacity=10000):
        self.storage = []
        self.capacity = capacity
        self.ptr = 0
        self.full = False

    def push(self, data):
        if self.full or len(self.storage) == self.capacity:
            self.full = True
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.capacity
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        obs, act, next_obs, reward, done = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            obs.append(np.array(X, copy=False))
            act.append(np.array(Y, copy=False))
            next_obs.append(np.array(U, copy=False))
            reward.append(np.array(R, copy=False))
            done.append(np.array(D, copy=False))

        return np.array(obs), np.array(act).reshape(-1, 1), np.array(next_obs), \
               np.array(reward).reshape(-1, 1), np.array(done).reshape(-1, 1)

    def is_full(self):
        return self.full

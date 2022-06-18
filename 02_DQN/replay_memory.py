import random
import collections
import numpy as np


class ReplayMemory:
    def __init__(self, max_size: int):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def to_array(self, data):
        return np.array(data).astype('float32')

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        for exp in mini_batch:
            s, a, r, s_p, done = exp
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return self.to_array(obs_batch), self.to_array(action_batch), \
            self.to_array(reward_batch), self.to_array(next_obs_batch), \
            self.to_array(done_batch)

    def __len__(self):
        return len(self.buffer)

"""
Functions that use multiple times
"""

import torch
import numpy as np


def to_dtype(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def train_one_episode(opt, wnet, gnet, done, next_obs, bs, ba, br, gamma):
    if done:
        v_s_ = 0.
    else:
        v_s_ = wnet(to_dtype(next_obs[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:   # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = wnet.loss_func(
        to_dtype(np.vstack(bs)),
        to_dtype(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else to_dtype(np.vstack(ba)),
        to_dtype(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(wnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # update worker net from global parameters
    wnet.load_state_dict(gnet.state_dict())


def record(global_ep, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1

    res_queue.put(ep_r)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % ep_r,
    )
# @Date    : 2022/3/23
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import random
import time

import cv2
import numpy as np
import gym
import torch

from algorithm import PPO

has_continuous_action_space = True
env_name = "Pendulum-v1"
env = gym.make(env_name)
# state space dimension
obs_dim = env.observation_space.shape[0]
# action space dimension
if has_continuous_action_space:
    act_dim = env.action_space.shape[0]
else:
    act_dim = env.action_space.n

################ hyperparameters ################
max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps
max_ep_len = 1000  # max timesteps in one episode
update_timestep = max_ep_len * 4  # update policy every n timesteps
K_epochs = 80  # update policy for K epochs in one PPO update

eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network

random_seed = 0  # set random seed if required (0 = no random seed)
action_std = 0.6  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)

save_model_freq = int(1e5)  # save model frequency (in num timesteps)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
checkpoint_path = "./models/PPO_{}.ckpt"
#####################################################
torch.manual_seed(random_seed)
env.seed(random_seed)
np.random.seed(random_seed)

if __name__ == '__main__':
    SHOW = False

    # initialize a PPO agent
    ppo_agent = PPO(obs_dim, act_dim, lr_actor, lr_critic, gamma,
                    K_epochs, eps_clip, has_continuous_action_space, action_std)

    if not SHOW:
        timestep = 0
        # training loop
        while timestep <= max_training_timesteps:

            state = env.reset()
            total_reward = 0

            for t in range(1, max_ep_len + 1):
                # select action with policy
                action = ppo_agent.sample_save_action(state)
                state, reward, done, _ = env.step(action)

                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                total_reward += reward
                timestep += 1
                # update PPO agent
                if timestep % update_timestep == 0:
                    ppo_agent.update()

                # if continuous action space; then decay action std of output action distribution
                if has_continuous_action_space and timestep % action_std_decay_freq == 0:
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                # save model weights
                if timestep % save_model_freq == 0:
                    print(f'episode: {timestep}, reward: {total_reward}')
                    ppo_agent.save(checkpoint_path.format(timestep))
                # break; if the episode is over
                if done:
                    break

    else:
        ppo_agent.load(checkpoint_path.format('3000000'))
        obs = env.reset()

        # 增加难度
        for _ in range(6):
            act = random.randint(-2, 2)
            act = np.array([act], dtype=np.float32)
            obs, _, _, _ = env.step(act)

        while True:
            action = ppo_agent.predict_action(obs)
            obs, reward, done, _ = env.step(action)

            time.sleep(0.05)
            cv2.imshow('demo', env.render(mode='rgb_array'))
            cv2.waitKey(1)
            if done:
                cv2.destroyAllWindows()
                break
        env.close()

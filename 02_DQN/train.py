import gym
import numpy as np

from model import Model
from algorithm import DQN
from agent import Agent
from replay_memory import ReplayMemory

LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率

MEMORY_SIZE = 200000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 64  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来

LEARNING_RATE = 0.0005  # 学习率
GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等


def train_episode(agent, env, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0

    while True:
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        rpm.append(
            (obs, action, reward, next_obs, done)
        )

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):  # 溢出
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = \
                rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(obs_batch, action_batch, reward_batch,
                                     next_obs_batch, done_batch)

        total_reward += reward
        obs = next_obs
        step += 1
        if done:
            break
    return total_reward


def evaluate_episode(agent, env, render):
    eval_reward = []
    for _ in range(5):
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
            env.close()
            cv2.destroyAllWindows()
            print(episode_reward)
            break


if __name__ == '__main__':
    SHOW = True

    game = 'MountainCar-v0'
    # CartPole-v1: expected reward > 475
    # MountainCar-v0 : expected reward > -120
    env = gym.make(game)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    save_path = f'./models/DQN-{game}.ckpt'

    rpm = ReplayMemory(MEMORY_SIZE)  # DQN经验回放池

    # 构建 agent
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    algorithm = DQN(model=model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm=algorithm,
        act_dim=act_dim,
        e_greed=0.1,
        e_greed_dec=1e-6
    )

    if SHOW:
        agent.load(save_path)
        show(agent, env)
    else:
        # 预存经验池
        while len(rpm) < MEMORY_WARMUP_SIZE:
            train_episode(agent, env, rpm)

        max_episode = 2000
        episode = 0
        while episode < max_episode:
            for _ in range(50):
                train_reward = train_episode(agent, env, rpm)
                episode += 1

            eval_reward = evaluate_episode(agent, env, render=False)
            print(f'episode: {episode:0>3d}, eval reward: {eval_reward}')

        agent.save(save_path)

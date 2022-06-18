# @Date    : 2022/3/18
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import numpy as np
import torch
import gym

from models import Model
from algorithm import AC
from agent import Agent


def train_episode(agent, env):
    obs_list, action_list, reward_list, next_obs_list = [], [], [], []
    obs = env.reset()

    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        next_obs, reward, done, _ = env.step(action)
        next_obs_list.append(next_obs)
        reward_list.append(reward)

        obs = next_obs

        if done:
            break
    return obs_list, action_list, reward_list, next_obs_list


def eval_episode(agent, env):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict_act(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def show(agent, env):
    import cv2

    obs = env.reset()
    episode_reward = 0

    while True:
        action = agent.predict_act(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

        cv2.imshow('demo', env.render(mode='rgb_array'))
        cv2.waitKey(1)

        if done:
            cv2.destroyAllWindows()
            env.close()
            print(episode_reward)
            break


if __name__ == '__main__':
    SHOW = False

    save_path = './models/AC.ckpt'

    # CartPole-v1: expected reward > 475
    env = gym.make('CartPole-v1')
    env.seed(1)
    torch.manual_seed(1)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    AC_alg = AC(model=model, lr=0.01)
    agent = Agent(alg=AC_alg)

    if SHOW:
        agent.load(save_path)
        show(agent, env)
    else:
        for episode in range(2000):
            obs_list, action_list, reward_list, next_obs_list = train_episode(agent, env)
            if episode % 10 == 0:
                print(f'Episode: {episode:0>3d}, Reward: {sum(reward_list)}')


            loss = agent.learn(reward_list)
            if (episode + 1) % 100 == 0:
                total_reward = eval_episode(agent, env)
                print(f'Test Reward: {total_reward}')
        agent.save(save_path)

"""
Episode: 000, Reward: 12.0
5.82363224029541
5.200126647949219
6.4244279861450195
23.61212921142578
10.382997512817383
5.910652160644531
5.828464031219482
5.698168754577637
3.4696273803710938
9.554821014404297
Episode: 010, Reward: 34.0
8.599145889282227
4.234831809997559
3.0924034118652344
11.86177921295166
5.899825572967529
13.121199607849121
3.052158832550049
5.491990089416504
8.064339637756348
5.755649089813232
Episode: 020, Reward: 20.0
7.197973251342773
26.9967041015625
9.174181938171387
14.211746215820312
4.297110080718994
5.8967695236206055
14.278529167175293
16.507099151611328
9.206096649169922
14.46363639831543
Episode: 030, Reward: 98.0
36.81689453125
5.7237653732299805
24.482358932495117
38.763187408447266
5.22113037109375
7.212644577026367
24.174232482910156
2.9068758487701416
10.214326858520508
12.720600128173828
Episode: 040, Reward: 31.0
11.204802513122559
7.19488000869751
9.379201889038086
7.7774658203125
7.146017074584961
11.114591598510742
11.823039054870605
7.497339725494385
20.49309539794922
9.738656044006348
Episode: 050, Reward: 32.0
13.846182823181152
27.421844482421875
11.672176361083984
21.337682723999023
6.758004665374756
23.345766067504883
10.421991348266602
8.93260383605957
8.588173866271973
5.7153425216674805
Episode: 060, Reward: 23.0
8.260984420776367
1.8904261589050293
8.779101371765137
12.660025596618652
16.067768096923828
10.298417091369629
12.009078025817871
27.0469913482666
8.253588676452637
13.381210327148438
Episode: 070, Reward: 48.0
14.583906173706055
5.789643287658691
12.954063415527344
23.4366512298584
5.454449653625488
7.231251239776611
12.988752365112305
9.253910064697266
16.129528045654297
9.44395923614502
Episode: 080, Reward: 34.0
9.731974601745605
44.85563278198242
2.4921045303344727
3.1534481048583984
15.947526931762695
2.535069465637207
3.721440315246582
205.2596435546875
112.87709045410156
0.43824005126953125
Episode: 090, Reward: 130.0
125.44774627685547
35.60887145996094
44.7290153503418
12.725055694580078
-4.442172050476074
19.01734733581543
-9.741519927978516
-11.01226806640625
-3.546154022216797
17.326141357421875
Test Reward: 149.4
"""

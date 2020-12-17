import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import os
import gym
import minerl
from model import *
from util import *

import pandas as pd
import numpy as np

def converter(observation):
    obs = observation['pov']
    obs = obs / 255.0
    compass_angle = observation['compassAngle']

    compass_angle_scale = 180
    compass_scaled = compass_angle / compass_angle_scale
    compass_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=obs.dtype) * compass_scaled
    obs = np.concatenate([obs, compass_channel], axis=-1)
    obs = torch.from_numpy(obs).float().to(device=device)
    obs = obs.permute(2, 0, 1)

    return obs


def main():

    total_episodes = 1000

    root_path = os.curdir
    model_path = root_path + '/dqn_model/'

    env = gym.make('MineRLNavigateDense-v0')
    checkpoint = torch.load(model_path+'model_weight_navigaate ver6.pth')
    env.make_interactive(port=6666, realtime=True)
    policy_net = DQN().to(device=device)
    policy_net.load_state_dict(checkpoint['model_state_dict'])

    total_steps = 0
    Episodes = [x for x in range(total_episodes)]
    eval_stats = pd.DataFrame(index=Episodes,
                               columns=['Mean Reward', 'Total Reward'])
    for num_epi in range(total_episodes):
        total_loss = 0
        obs = env.reset()
        state = converter(obs)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            steps += 1
            total_steps += 1

            a_out = policy_net.sample_action(state, 0.05)
            #print(a_out)
            #action_index = torch.argmax(a_out, dim=1)[0]

            # Action들을 정의
            action = env.action_space.noop()
            if (a_out == 0):
                action['forward'] = 1
            elif (a_out == 1):
                action['camera'] = [0, 5]
            elif (a_out == 2):
                action['camera'] = [0, -5]
            # elif (action_index == 3):
            #     action['camera'] = [5, 0]
            # elif (action_index == 4):
            #     action['camera'] = [-5, 0]

            action['jump'] = 1
            action['attack'] = 1
            obs_prime, reward, done, info = env.step(action)
            state_prime = converter(obs_prime)

            state = state_prime
            total_reward += reward

            if done:
                print("%d episode is done" % num_epi)
                #writer.add_scalar('total reward', total_reward / steps, num_epi)
                eval_stats.loc[num_epi]['Mean Reward'] = total_reward / steps
                eval_stats.loc[num_epi]['Total Reward'] = total_reward

                eval_stats.to_csv('eval_stat mineRL Agent 3 actions ver7.csv')
                break

            if steps % 10 == 0:
                #print("total loss : ", total_loss)
                print("steps : ", steps)
                print("total rewards : ", total_reward)

    #writer.close()
    env.close()
if __name__ == "__main__":
    main()
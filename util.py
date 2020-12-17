import numpy as np
import torch
from collections import deque
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.device_count()
print(torch.cuda.get_device_name(0))
torch.cuda.device(0)
cuda = torch.device('cuda')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

buffer_limit = 100000
def converter(observation):
    obs = observation['pov']
    #obs = obs / 255.0
    compass_angle = observation['compassAngle']

    compass_angle_scale = 180
    compass_scaled = compass_angle / compass_angle_scale
    compass_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=obs.dtype) * compass_scaled
    obs = np.concatenate([obs, compass_channel], axis=-1)
    obs = torch.from_numpy(obs).float().to(device=device)
    obs = obs.permute(2, 0, 1)
    return obs


class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_list = []
        action_list = []
        reward_list =[]
        next_state_list = []
        done_mask_list = []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            state_list.append(s)
            action_list.append([a])
            reward_list.append([r])
            next_state_list.append(s_prime)
            done_mask_list.append([done_mask])

        a = state_list
        b = torch.tensor(action_list, dtype=torch.int64)
        c = torch.tensor(reward_list)
        d = next_state_list
        e = torch.tensor(done_mask_list)

        return a, b, c, d, e

    def size(self):
        return len(self.buffer)
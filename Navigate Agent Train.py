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

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

writer = SummaryWriter('runs/MineRLAgent_experiment_14')

torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.device_count()
print(torch.cuda.get_device_name(0))
torch.cuda.device(0)
cuda = torch.device('cuda')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



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


class DQN(nn.Module):
    def __init__(self):
        self.num_actions = 3
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(64, 8, 4)
        convw = conv2d_size_out(convw, 4, 2)
        convw = conv2d_size_out(convw, 3, 1)

        linear_input_size = convw * convw * 64
        self.head = nn.Linear(linear_input_size, self.num_actions)

    def forward(self, x):
        if(len(x.shape) < 4):
            x = x.unsqueeze(0).to(device=device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.head(x.view(x.size(0), -1)))  # view는 numpy의 reshape 와 같다.
        x = F.softmax(x, dim=1)
        return x

def train(q, q_target, memory, optimizer, num_epi, total_loss):
    for i in range(10):
        sample_batch = memory.sample(batch_size)
        s = torch.stack(sample_batch[0]).float().to(device)
        a = sample_batch[1].to(device)
        r = sample_batch[2].float().to(device)
        s_prime = torch.stack(sample_batch[3]).float().to(device)
        done_mask = sample_batch[4].float().to(device)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        end_multiplier = -(done_mask - 1)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * end_multiplier
        loss = F.smooth_l1_loss(q_a, target).to(device)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, total_loss

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

#하이퍼 파라미터
learning_rate = 0.001
gamma = 0.99
buffer_limit = 75000
batch_size = 256

def main():

    total_episodes = 100
    startEpsilon = 1.0
    endEpsilon = 0.1
    epsilon = startEpsilon

    root_path = os.curdir
    model_path = root_path + '/dqn_model/'

    Episodes = [x for x in range(total_episodes)]
    train_stats = pd.DataFrame(index=Episodes,
                               columns=['Train loss', 'Mean Reward', 'Total Reward'])


    stepDrop = (startEpsilon - endEpsilon) / total_episodes

    env = gym.make('MineRLNavigateDense-v0')
    env.make_interactive(port=6666, realtime=False)
    policy_net = DQN().to(device=device)
    target_net = DQN().to(device=device)
    target_net.load_state_dict(policy_net.state_dict())
    memory = ReplayBuffer()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    print_interval = 10
    total_steps = 0

    for num_epi in range(total_episodes):
        #checkpoint = torch.load(model_path+'model_weight_treechop.pth')
        #num_epi = checkpoint['num_epi']
        #epsilon = checkpoint['epsilon']
        #print(num_epi)
        #print(torch.load(model_path+'model_weight.pth'))
        #policy_net.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


        total_loss = 0
        obs = env.reset()
        state = converter(obs)
        done = False
        total_reward = 0
        steps = 0

        if(epsilon > endEpsilon):
            epsilon -= stepDrop

        while not done:
            steps += 1
            total_steps += 1

            if np.random.rand(1) < epsilon:
                action_index = np.random.randint(0,3)
            else:
                a_out = policy_net.forward(state)
                if steps % 100 == 0:
                    print(a_out)
                action_index = torch.argmax(a_out)
                #action_index = torch.argmax(a_out, dim=1)[0]

            # Action들을 정의
            action = env.action_space.noop()
            if (action_index == 0):
                action['forward'] = 1
            elif (action_index == 1):
                action['camera'] = [0, 5]
            elif (action_index == 2):
                action['camera'] = [0, -5]
            # elif (action_index == 3):
            #     action['camera'] = [5, 0]
            # elif (action_index == 4):
            #     action['camera'] = [-5, 0]

            action['jump'] = 1
            action['attack'] = 1
            obs_prime, reward, done, info = env.step(action)
            state_prime = converter(obs_prime)

            memory.put((state, action_index, reward, state_prime, done))
            state = state_prime
            total_reward += reward

            if done:
                print("%d episode is done" % num_epi)
                torch.save({'num_epi': num_epi, 'model_state_dict' : policy_net.state_dict(), 'optimizer_state_dict' : optimizer.state_dict(),
                            'loss' : total_loss, 'epsilon' : epsilon}, model_path + 'model_weight_navigaate ver4.pth')
                print("model saved")
                writer.add_scalar('train_loss', total_loss / steps, num_epi)
                writer.add_scalar('total reward', total_reward / steps, num_epi)
                train_stats.loc[num_epi]['Train loss'] = total_loss / steps
                train_stats.loc[num_epi]['Mean Reward'] = total_reward / steps
                train_stats.loc[num_epi]['Total Reward'] = total_reward

                train_stats.to_csv('train_stat mineRL Agent 6 actions ver4.csv')
                break

            temp_loss = 0
            if memory.size() > 5000:
                temp_loss, total_loss = train(policy_net, target_net, memory, optimizer, num_epi, total_loss)
            if memory.size() > 75000:
                memory.buffer.popleft()

            if steps % 100 == 0:
                print("loss : ", temp_loss)
                print("total loss : ", total_loss)
                print("total rewards : ", total_reward)
        if num_epi % 10 == 0 and num_epi != 0:
            # 특정 반복 수가 되면 타겟 네트워크도 업데이트
            print("target network updated")
            target_net.load_state_dict(policy_net.state_dict())
            print("n_episode :{}, n_buffer : {}, eps : {:.1f}%".format(num_epi, memory.size(), epsilon * 100))

    writer.close()
    env.close()

if __name__ == "__main__":
    main()
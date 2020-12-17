import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import gym
import minerl

from util import *
from model import *
import pandas as pd

writer = SummaryWriter('runs/MineRLAgent_experiment_19')

#하이퍼 파라미터
learning_rate = 0.001
gamma = 0.99
buffer_limit = 100000
batch_size = 64


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
        loss = F.mse_loss(q_a, target).to(device)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, total_loss

def main():

    total_episodes = 1000
    startEpsilon = 1.0
    endEpsilon = 0.05
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
        temp_loss = 0
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

            a_out = policy_net.sample_action(state,epsilon)
            if steps % 100 == 0:
                print(a_out)
            action_index = a_out
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
                            'loss' : total_loss, 'epsilon' : epsilon}, model_path + 'model_weight_navigaate ver6.pth')
                print("model saved")
                writer.add_scalar('train_loss', temp_loss, num_epi)
                writer.add_scalar('total reward', total_reward / steps, num_epi)
                train_stats.loc[num_epi]['Train loss'] = temp_loss.item()
                train_stats.loc[num_epi]['Mean Reward'] = total_reward / steps
                train_stats.loc[num_epi]['Total Reward'] = total_reward

                train_stats.to_csv('train_stat mineRL Agent 6 actions ver6.csv')
                break


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
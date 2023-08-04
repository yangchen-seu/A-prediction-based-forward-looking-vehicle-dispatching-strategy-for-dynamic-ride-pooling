'''
Author: your name
Date: 2021-12-05 16:23:47
LastEditTime: 2021-12-27 22:28:41
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \强化学习教程\common\model.py
'''
from typing import ForwardRef
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class MLP(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, n_actions) # 输出层

    def forward(self, x):
        # 各层的激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        
class DQN(nn.Module):
    
    
    def __init__(self, n_states, n_actions, location_dim, hidden_dim = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # 隐藏层
        
        self.embedding_layer_output_dim = 2 * int(np.floor(np.power(location_dim, 0.25)))
        self.embedding_layer1 = nn.Linear(location_dim, hidden_dim) 
        self.embedding_layer2 = nn.Linear(hidden_dim, self.embedding_layer_output_dim)

        self.fc3 = nn.Linear(hidden_dim + self.embedding_layer_output_dim, n_actions) # 输出层

    def forward(self, x, embedding_info):
        # 各层的激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        embedding_info = F.softmax(self.embedding_layer1(embedding_info))
        embedding_info = F.softmax(self.embedding_layer2(embedding_info))
        x = torch.cat([x, embedding_info], 1)

        return self.fc3(x)


class critic(nn.Module):
    def __init__(self, n_obs, hidden_size, init_w = 3e-3) -> None:
        super().__init__()
        self.l1 = nn.Linear(n_obs, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)
        # 随机初始化为较小的值
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)

    
    def forward(self, state):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class Actor(nn.Module):
    def __init__(self, n_obs, n_actions, hidden_size, init_w = 3e-3) -> None:
        super().__init__()
        self.l1 = nn.Linear(n_obs, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, n_actions)

        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x

class ActorCritic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim = 256) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim = 1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return value, dist

        
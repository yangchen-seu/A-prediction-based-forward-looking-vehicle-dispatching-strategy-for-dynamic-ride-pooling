
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np




from common.model import critic

class Critic:
    

    def __init__(self, n_state, cfg):

        self.device = cfg.device  # 设备，cpu或gpu等
        self.gamma = cfg.gamma  # 奖励的折扣因子
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = 0.8

        self.value = 0 # 采取的动作

        self.value_net = critic(n_state, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = critic(n_state, hidden_dim=cfg.hidden_dim).to(self.device)

        for target_param, param in zip(self.target_net.parameters(),self.value_net.parameters()): # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=cfg.lr) # 优化器


    def update(self, memory):
        if len(memory) < self.batch_size: # 当memory中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state_batch, reward_batch, next_state_batch, done_batch = memory.sample(
            self.batch_size)
        # 转为张量
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)  
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)

        values = self.value_net(state_batch) # 计算当前状态(s_t)对应的V(s_t)
        next_values = self.target_net(next_state_batch) # 计算下一时刻的状态值
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_values = reward_batch + self.gamma * next_values * (1-done_batch)

        loss = nn.MSELoss()(values, expected_values.unsqueeze(1))  # 计算均方根损失
        # 优化更新模型
        self.optimizer.zero_grad()  

        loss.backward()
        for param in self.value_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 


    def save(self, path):
        torch.save(self.target_net.state_dict(), path+'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path+'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
'''
Author: your name
Date: 2021-12-05 16:23:54
LastEditTime: 2021-12-05 16:34:31
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \强化学习教程\common\memory.py
'''
import random
class ReplayBuffer:
    def __init__(self, capacity) -> None:
        self.capacity = capacity # 容量
        self.buffer = [] # 缓冲区
        self.posotion = 0

    def push(self, state, action, reward, next_state, done):
        # 缓冲区是一个队列，容量超出限制时，除去开始存入的transition
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.posotion] = (state, action, reward, next_state, done)
        self.posotion = (self.posotion + 1) % self.capacity

    def sample(self, batch_size):
        # 取数据
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
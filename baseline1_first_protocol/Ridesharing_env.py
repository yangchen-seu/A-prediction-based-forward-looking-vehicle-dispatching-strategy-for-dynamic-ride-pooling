'''
Author: your name
Date: 2022-02-21 14:36:55
LastEditTime: 2022-02-21 16:10:36
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \强化学习网约车\random strategy\Ridesharing_env.py
'''

import gym
from gym import spaces
import numpy as np
import Control
# from Reinforcementlearning.Config import Config


class Ridesharing_env(gym.Env):
    """
    The training environment for vehicle dispatching.
    """
    def __init__(self, agent_lis):
        for agent in agent_lis:
            agent.reset()

        self.control = Control.Control(agent_lis)
        self.render = False
         # timesteps
        self.episode_timestep = 0
        self.n_episode        = 0

    def step(self, agent_lis):
        # print('CustomEnv Step successful!')
        reward, done = self.control.step(agent_lis)
        if self.render:
            self.show()
        return reward, done

    def reset(self, agent_lis):
        self.__init__(agent_lis)


    def show(self):
        pass


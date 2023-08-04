import os
import sys

import gym
from gym import spaces
import numpy as np
from ... import Control
from ... import Network

class Ridesharing_env(gym.Env):
    """
    The training environment for vehicle dispatching.
    """
    def __init__(self,time_length,lon_max,lon_min,lat_max,lat_min):
        self.control = Control.Control(time_length,lon_max,lon_min,lat_max,lat_min)
        self.render = False
         # timesteps
        self.episode_timestep = 0
        self.n_episode        = 0

        # # action space
        # self.n_plans = 3 # pickup_general_order pickup_ridesharing_order rebalance

        # self.action_space = spaces.Discrete(self.n_plans)

        # # observation space
        # obs_demand = np.zeros((50,50)) # 第一层，周边50个网格的需求分布
        # obs_supply = np.zeros((50,50)) # 第二层，周边50个网格的供给分布
        # obs_price = np.ones((50,50)) # 第三层，周边50个网格的价值分布

        # self.observation_space = spaces.Discrete(
        #     obs_demand, obs_supply, obs_price, dtype=np.float32
        # )

    def step(self, agent):
        print('CustomEnv Step successful!')
        state, reward, done = self.control.step(agent)
        if self.render:
            self.show()
        return state, reward, done

    def reset(self):
        self.__init__()
        print('CustomEnv Environment reset')

    def show(self):
        pass

    # def get_vehicle_state(self, VehicleAgent):
    #     '''
    #     description: 计算每个智能体的状态
    #     param {
    #         VehicleAgent
    #     }
    #     return {
    #         list: states of each agent
    #     }
    #     '''     
    #     # position of each VehicleAgent
    #     position = VehicleAgent.location   
    #     passengers = VehicleAgent.passengers
    #     return [position, passengers]


    # @property
    # def state(self):
    #     s = {}
    #     for v in self.vehicles:
    #         s[v] = self.get_vehicle_state(v)
    #     return s

    # def step_vehicle(self, v, action):
    #     assert self.action_space.contains(action), f"Invalid action: {action}"
    #     action_path = v.plans[action]
    #     v.set_route(action_path)


    # # pickup action
    # def pickup(self, vehicle):
    #     state = vehicle.get_state(self.current_time)
    #     if state == 0:
    #         # 人满了，不能接客，沿原定路径继续行驶
    #         return vehicle.path
    #     elif state == 1:
    #         # 可以接订单
    #         order = self.find_order(vehicle.location) # 返回距离最近的订单
    #         self.vehicle_list[i].action_for_order(order)
    #     else:
    #         # 有选择性的接单
    #         order = self.find_avaliable_order(self.vehicle_list[i])
    #         if order:
    #             self.vehicle_list[i].action_for_order(order)


    # # rebalance
    # def rebalance():
    #     pass

    # # donothing
    # def donothing():
    #     pass

    # def step(self, actions):
    #     """所有智能体执行动作。

    #     Params
    #     ------
    #     actions: 智能体动作向量。

    #     Returns
    #     -------
    #     state: 状态向量。
    #     reward: 奖励值。
    #     terminal: 本轮是否终止。
    #     info: 备用信息。
    #     """
    #     self.reload_sumo()
    #     # select action for each vehicle
    #     for v, a in zip(self.vehicles, actions):
    #         # print(f'===========step: {type(v), v}')
    #         self.step_vehicle(self.vehicles[v], a)

    #     start_trip = [False] * len(self.vehicles)
    #     finish_trip = [False] * len(self.vehicles)
    #     trip_travel_times = [0] * len(self.vehicles)
    #     # print(f'=========== validate {not all(finish_trip)}')
    #     # run simulation until all trips finish
    #     while not all(finish_trip):
    #         for i, v in enumerate(self.vehicles):
    #             if not finish_trip[i]:
    #                 current_edge = self.vehicles[v].current_edge
    #                 if not start_trip[i]:
    #                     if current_edge == self.vehicles[v].origin:
    #                         start_trip[i] = True
    #                 else:
    #                     trip_travel_times[i] += 1 # count travel time
    #                     if current_edge == self.vehicles[v].destination:
    #                         finish_trip[i] = True

    #     # update memories
    #     for i, v in enumerate(self.vehicles):
    #         # print(len(actions))
    #         # print(len(trip_travel_times))
    #         # print(len(self.vehicles))
    #         self.memory_travel_times[v].update(actions[i], trip_travel_times[i])

    #     self.episode_timestep += 1

    #     reward = self.compute_reward(trip_travel_times, actions)
    #     terminal = self.is_terminal()
    #     info = {}
    #     return self.state, reward, terminal, info


    # def compute_reward(self, travel_times, actions):
    #     """计算奖励。

    #     Returns
    #     -------
    #     reward: [list] 所有车辆的奖励。
    #     """
    #     reward = []
    #     for i, v in enumerate(self.vehicle_ids_RL):
    #         travel_time = travel_times[i]
    #         action = actions[i]

    #         rank = self.memory_travel_times[v].action_rank(action)
    #         rank = 1 - rank

    #         actual_speed = get_path_length(self.plans[v][action]) / travel_time
    #         actual_speed *= 1000
    #         speed_diff = actual_speed - 50 / 3.6
    #         speed_diff /= 20
    #         reward += [rank + speed_diff]
    #     return reward
    

    # def is_terminal(self):
    #     if self.episode_timestep >= CFG.env.max_episode_timesteps:
    #         return 1
    #     else:
    #         return 0



    # def set_seed(self, seed=CFG.env.random_state):
    #     self.random_state = seed
    #     self.action_space.seed(seed)
    #     self.seed(seed)


    # def reset(self):
    #     obs = self.get_observation()
    #     return obs

    # def render(self):
    #     # 可视化环境
    #     pass

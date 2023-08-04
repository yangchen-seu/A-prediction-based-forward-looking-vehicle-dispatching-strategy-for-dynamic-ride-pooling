'''
Author: your name
Date: 2021-12-06 22:47:09
LastEditTime: 2022-03-27 15:47:25
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \强化学习网约车\Agent.py
'''

import time



class Vehicle:
    

    def __init__(self, driver_id, location, cfg):
        self.cfg = cfg
        self.id = driver_id
        self.passengers = 0 # 0 没有乘客，1 可接一个拼车乘客，2 不可接乘客
        self.order_list = [] # 当前正在响应的订单
        self.his_order_list = [] # 响应过的所有订单
        self.reward = 0
        self.target = 0
        self.trips_1 = [] # 可响应的单人trips
        self.trips_2 = [] # 可响应的拼车trips
        self.trips = [] # 可相应的trips
        self.e_r_v = []

        self.reposition_target = 1 # 没有订单可接，进入下一次匹配
        self.p0_pickup_distance = 0
        self.drive_distance = 0
        
        self.location = location # 当前智能体的位置
        self.path = []
        self.path_length = []
        self.origin_location = location # 拼车前的位置
        self.destination = 0 # 当前智能体的目的地
        self.state = 0 # 能否执行动作
        self.activate_time  = time.mktime(time.strptime(cfg.date + cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))

        self.action = 0 # 采取的动作

    # 判断司机是否能执行动作
    def is_activate(self, time):
        if time > self.activate_time:
            self.state = 1
        else:
            self.state = 0
    # 重置reposition 指令
    def reset_reposition(self):
        self.reposition_target = 0

        
    # 重置
    def reset(self):
        self.passengers = 0 # 0 没有乘客，1 可接一个拼车乘客，2 不可接乘客
        self.order_list = [] # 当前正在响应的订单
        self.his_order_list = [] # 响应过的所有订单
        self.reward = 0

        self.path = [] # 当前的行驶路径
        self.state = 0 # 能否执行动作
        self.activate_time  = time.mktime(time.strptime(self.cfg.date + self.cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))

        self.action = 0 # 采取的动作

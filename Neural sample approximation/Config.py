'''
Author: your name
Date: 2021-12-07 21:36:06
LastEditTime: 2022-02-21 16:42:55
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \强化学习网约车\Reinforcementlearning\Config.py
'''

class Config:

    def __init__(self) -> None:
        self.vehicle_num = 100

        # files
        self.input_path = 'input\\'
        self.output_path = 'output\\'
        
        self.order_file_name = 'order.csv'
        self.network_file = 'network.csv'
        self.shortest_path_file = 'shortest_path.csv'

        # 与网络相关的参数
        self.R = 2 # 搜索半径
        self.unit_distance_value = 10 # 平台收益一公里5块钱
        self.unit_distance_cost = 2 # 司机消耗一公里2块钱
        
        self.date = '2017-05-01'
        self.simulation_begin_time = ' 08:00:00' # 仿真开始时间
        self.simulation_end_time = ' 09:00:00' # 仿真结束时间
        self.unit_driving_time = 120/1000 # 行驶速度
        self.unit_time_value = 1.5/120 # 每秒的行驶费用
        self.demand_ratio = 1

        # matching condition
        self.pickup_distance_threshold = 2000
        self.extra_distance_threshold = 3000
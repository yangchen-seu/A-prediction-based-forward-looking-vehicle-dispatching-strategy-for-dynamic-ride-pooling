'''
Author: your name
Date: 2021-12-07 21:36:06
LastEditTime: 2023-01-10 01:23:40
LastEditors: yangchen-seu 58383528+yangchen-seu@users.noreply.github.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \强化学习网约车\Reinforcementlearning\Config.py
'''

class Config:

    def __init__(self) -> None:
        self.device = 'cpu'
        # files
        self.input_path = 'input\\'
        self.output_path = 'output\\'
        
        self.order_file= '../input/order.csv'
        self.shortest_path_file = '../input/shortest_path.csv'

        # 与网络相关的参数
        self.unit_distance_value = 5 # 平台收益一公里5块钱
        self.unit_distance_cost = 2 # 司机消耗一公里2块钱
        
        self.date = '2017-05-01'
        self.simulation_begin_time = ' 08:00:00' # 仿真开始时间
        self.simulation_end_time = ' 09:00:00' # 仿真结束时间
        self.unit_driving_time = 120/1000 # 行驶速度
        self.unit_time_value = 1.5/120 # 每秒的行驶费用
        self.demand_ratio = 0.3
        self.order_driver_ratio = 100 / 25 
        self.progress_target = True
        self.time_unit = 10

        self.optimazition_target = 'expected_saved_distance' # platform_income, expected_saved_distance,combination
        self.matching_condition = True
        # matching condition
        self.pickup_distance_threshold = 3000
        self.detour_distance_threshold = 3000
        self.delay_time_threshold = 90
        self.dead_value = -1e4
        self.reposition_time_threshold = 120
        self.discount_factor = 0.7
        self.rw = 0.8
        self.beta = 0.6
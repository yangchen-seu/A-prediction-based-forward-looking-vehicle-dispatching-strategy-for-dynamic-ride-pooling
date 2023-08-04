
import Config
import pandas as pd
import numpy as np
import Network as net
import Seeker
import random
import Vehicle
import os
from pulp import *
import time


class Simulation():

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.date = cfg.date
        # cfg.demand_ratio = 0.01
        self.order_list = pd.read_csv(self.cfg.order_file).sample(
            frac=cfg.demand_ratio, random_state=1)
        self.vehicle_num = int(len(self.order_list) / cfg.order_driver_ratio)
        self.order_list['beginTime_stamp'] = self.order_list['dwv_order_make_haikou_1.departure_time'].apply(
            lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        self.begin_time = time.mktime(time.strptime(
            cfg.date + cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))
        self.end_time = time.mktime(time.strptime(
            cfg.date + cfg.simulation_end_time, "%Y-%m-%d %H:%M:%S"))
        self.locations = self.order_list['O_location'].unique()
        self.network = net.Network()
        self.shortest_path = pd.read_csv(self.cfg.shortest_path_file)

        self.time_unit = 10  # 控制的时间窗,每10s匹配一次
        self.index = 0  # 计数器
        self.device = cfg.device
        self.total_reward = 0
        self.optimazition_target = cfg.optimazition_target  # 仿真的优化目标
        self.matching_condition = cfg.matching_condition  # 匹配时是否有条件限制
        self.pickup_distance_threshold = cfg.pickup_distance_threshold
        self.detour_distance_threshold = cfg.detour_distance_threshold
        self.vehicle_list = []
        self.visualization_df = pd.DataFrame(columns=[
                                             'vehicle_id', 'vehicle_type', 'depart_time', 'arrival_time', 'O_location', 'D_location', 'p1_id', 'p2_id'])
        self.takers = []
        self.current_seekers = []  # 存储需要匹配的乘客
        self.remain_seekers = []
        self.time_reset()

        for i in range(self.vehicle_num):
            random.seed(i)
            location = random.choice(self.locations)
            vehicle = Vehicle.Vehicle(i, location, self.cfg)
            self.vehicle_list.append(vehicle)
            # 重置司机的时间
            vehicle.activate_time = self.time

        # system metric
        self.his_order = []  # all orders responsed
        self.waitingtime = []
        self.detour_distance = []
        self.traveltime = []
        self.pickup_time = []
        self.platform_income = []
        self.shared_distance = []
        self.reposition_time = []
        self.total_travel_distance = 0
        self.saved_travel_distance = 0

        self.carpool_order = []
        self.ride_distance_error = []
        self.shared_distance_error = []
        self.relative_ride_distance_error = []
        self.relative_shared_distance_error = []

    def reset(self):
        self.takers = []
        self.current_seekers = []  # 存储需要匹配的乘客
        self.remain_seekers = []
        self.vehicle_list = []
        self.total_reward = 0
        self.order_list = pd.read_csv(self.cfg.order_file).sample(
            frac=self.cfg.demand_ratio, random_state=1)
        self.order_list['beginTime_stamp'] = self.order_list['dwv_order_make_haikou_1.departure_time'].apply(
            lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        self.begin_time = time.mktime(time.strptime(
            self.cfg.date + self.cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))
        self.end_time = time.mktime(time.strptime(
            self.cfg.date + self.cfg.simulation_end_time, "%Y-%m-%d %H:%M:%S"))
        self.order_list = self.order_list[self.order_list['beginTime_stamp']
                                          >= self.begin_time]
        self.order_list = self.order_list[self.order_list['beginTime_stamp']
                                          <= self.end_time]

        self.time_reset()

        for i in range(self.vehicle_num):
            random.seed(i)
            location = random.choice(self.locations)
            vehicle = Vehicle.Vehicle(i, location, self.cfg)
            self.vehicle_list.append(vehicle)
            # 重置司机的时间
            vehicle.activate_time = self.time

        # system metric
        self.his_order = []  # all orders responsed
        self.waitingtime = []
        self.detour_distance = []
        self.traveltime = []
        self.pickup_time = []
        self.platform_income = []
        self.shared_distance = []
        self.reposition_time = []
        self.total_travel_distance = 0
        self.saved_travel_distance = 0

        self.carpool_order = []
        self.ride_distance_error = []
        self.shared_distance_error = []
        self.relative_ride_distance_error = []
        self.relative_shared_distance_error = []
        self.visualization_df = pd.DataFrame(columns=[
                                             'vehicle_id', 'vehicle_type', 'depart_time', 'arrival_time', 'O_location', 'D_location', 'p1_id', 'p2_id'])

    def time_reset(self):
        # 转换成时间数组
        self.time = time.strptime(
            self.cfg.date + self.cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S")
        # 转换成时间戳
        self.time = time.mktime(self.time)
        self.time_slot = 0
        # print('time reset:', self.time)

    def step(self,):
        seekers = []
        for index, row in self.order_list.iterrows():
            seeker = Seeker.Seeker(row)
            seeker.set_shortest_path(self.get_path(
                seeker.O_location, seeker.D_location))
            value = self.cfg.unit_distance_value / 1000 * seeker.shortest_distance
            seeker.set_value(value)
            seekers.append(seeker)

        reward, done = self.process(seekers)

        return reward,  done

    #
    def process(self, seekers):

        reward = self.optimal_assignment(seekers)

        return reward,  False

    # 最优匹配算法
    def optimal_assignment(self,  seekers):
        start = time.time()
        dim = len(seekers)

        # 定义cost
        weight = np.ones((len(seekers), len(seekers))) * self.cfg.dead_value
        for i in range(len(seekers)):
            weight[i][i] = 0
            for j in range(i + 1, len(seekers)):
                weight[i][j] = self.cal_rr_weight(seekers[i], seekers[j])
                weight[j][i] = weight[i][j]

        # 定义问题
        prob = LpProblem("optimal_assignment", LpMaximize)

        # 定义0-1整数变量
        x = LpVariable.dicts("x", (range(dim), range(dim)), cat=LpBinary)

        # 添加目标函数
        objective = pulp.lpSum(weight[r][c] * x[r][c]
                               for r in range(dim) for c in range(dim))
        prob += objective

        # 添加约束条件
        for r in range(dim):
            prob += lpSum(x[r][c] for c in range(dim)) <= 1

        for c in range(dim):
            prob += lpSum(x[r][c] for r in range(dim)) <= 1

        # Add the constraint x[r][c] = x[c][r]
        for r in range(dim):
            for c in range(dim):
                prob += x[r][c] == x[c][r]

        # 求解问题
        prob.solve(PULP_CBC_CMD(msg=0))
        end = time.time()

        print('求解优化问题用时:', end - start)
        # Print the values of the variables
        matching_pair = {}
        row = []
        df = pd.DataFrame(columns=['p0_id', 'p0_begin_time', 'p0_O_location', 'p0_D_location',
                                   'p1_id', 'p1_begin_time', 'p1_O_location', 'p1_D_location'])
        for r in range(dim):
            for c in range(dim):
                if x[r][c].varValue == 1:
                    print(f"x[{r}][{c}] = {x[r][c].varValue}")
                    matching_pair[(r, c)] = (seekers[r], seekers[c])
                    row = [seekers[r].id, seekers[r].begin_time, seekers[r].O_location, seekers[r].D_location,
                           seekers[c].id, seekers[c].begin_time, seekers[c].O_location, seekers[c].D_location]

                    df.loc[len(df)] = row
        print('dim', dim)
        df.to_csv('output/optimal_matching_max_matching.csv')
        return

    def is_fifo(self, p0, p1):

        fifo = [self.get_path(p1.O_location, p0.D_location),
                self.get_path(p0.D_location, p1.D_location)]
        lifo = [self.get_path(p1.O_location, p1.D_location),
                self.get_path(p1.D_location, p0.D_location)]
        if sum(fifo) < sum(lifo):
            return True, fifo
        else:
            return False, lifo

    # 判断两个乘客是否可以组合成e(r,r)
    def cal_rr_weight(self, p0, p1):
        if abs(p0.begin_time_stamp - p1.begin_time_stamp) > 90:
            return self.cfg.dead_value
        if p0.begin_time_stamp > p1.begin_time_stamp:
            p0,p1 = p1,p0
        pick_up_distance = self.get_path(
            p1.O_location, p0.O_location)

        fifo, distance = self.is_fifo(p0, p1)
        if fifo:
            shared_distance = self.get_path(
                p1.O_location, p0.D_location)
            p0_invehicle = pick_up_distance + distance[0]
            p1_invehicle = sum(distance)
            p0_detour = p0_invehicle - p0.shortest_distance
            p1_detour = p1_invehicle - p1.shortest_distance

        else:
            shared_distance = p1.shortest_distance
            p0_invehicle = pick_up_distance + sum(distance)
            p1_invehicle = distance[0]
            p0_detour = p0_invehicle - p0.shortest_distance
            p1_detour = p1_invehicle - p1.shortest_distance

        distance_saving = p0.shortest_distance + p1.shortest_distance - \
            (p0_invehicle + p1_invehicle - shared_distance)

        if p0_detour < self.cfg.detour_distance_threshold and p1_detour < self.cfg.detour_distance_threshold:
            return  1 # distance_saving
        else:
            return self.cfg.dead_value

    def get_path(self, O, D):
        tmp = self.shortest_path[(self.shortest_path['O'] == O) & (
            self.shortest_path['D'] == D)]
        if tmp['distance'].unique():
            return tmp['distance'].unique()[0]
        else:
            return self.network.get_path(O, D)[0]

    def save_metric(self, path):
        import pickle

        with open(path, "wb") as tf:
            pickle.dump(self.res, tf)

    def save_his_order(self, path):
        dic = {}
        for i in range(len(self.his_order)):
            dic[i] = self.his_order[i]
        import pickle

        with open(path, "wb") as tf:
            pickle.dump(dic, tf)


cfg = Config.Config()
simu = Simulation(cfg)
simu.step()


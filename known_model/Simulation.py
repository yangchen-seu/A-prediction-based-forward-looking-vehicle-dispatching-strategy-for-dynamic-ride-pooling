
import time
import pandas as pd
import numpy as np
import Network as net
import Seeker
import random
from common import KM_method
import Vehicle
import os


class Simulation():

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.date = cfg.date
        self.order_list = pd.read_csv(self.cfg.order_file).sample(
            frac=cfg.demand_ratio, random_state=1)
        self.vehicle_num = int(len(self.order_list) / cfg.order_driver_ratio)
        self.order_list['beginTime_stamp'] = self.order_list['dwv_order_make_haikou_1.departure_time'].apply(
            lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        self.optimal_matching = pd.read_csv('output/snw0501_sharedistance_530_90.csv')
        # self.optimal_matching = pd.read_csv('output/optimal_matching_max_matching.csv')
        p0_id = self.optimal_matching['p0_id']
        p1_id = self.optimal_matching['p1_id']
        self.matched_results = {}
        for i in range(len(p0_id)):
            self.matched_results[p0_id[i]] = p1_id[i]
            self.matched_results[p1_id[i]] = p0_id[i]
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

        self.current_seekers = []  # 存储需要匹配的乘客
        self.Seekers = {}
        for index, row in self.order_list.iterrows():
            seeker = Seeker.Seeker(row)
            seeker.set_shortest_path(self.get_path(
                seeker.O_location, seeker.D_location))
            # 生成一个正态分布随机数
            # 设置随机数生成的种子
            np.random.seed(index)
            seeker.waitingtime_threshold = max(30, np.random.normal(self.cfg.delay_time_threshold, 10))
            value = self.cfg.unit_distance_value / 1000 * seeker.shortest_distance
            self.Seekers[seeker.id] = seeker
            seeker.set_value(value)

        self.remain_seekers = []
        self.activate_drivers = 0
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

    def time_reset(self):
        # 转换成时间数组
        self.time = time.strptime(
            self.cfg.date + self.cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S")
        # 转换成时间戳
        self.time = time.mktime(self.time)
        self.time_slot = 0
        # print('time reset:', self.time)

    def step(self,):
        time_old = self.time
        self.time += self.time_unit
        self.time_slot += 1

        # 筛选时间窗内的订单
        current_time_orders = self.order_list[self.order_list['beginTime_stamp'] > time_old]
        current_time_orders = current_time_orders[current_time_orders['beginTime_stamp'] <= self.time]
        self.current_seekers = []
        self.current_seekers_location = []
        for index, row in current_time_orders.iterrows():
            seeker_id = row['dwv_order_make_haikou_1.order_id']
            seeker = self.Seekers[seeker_id] 
            if seeker not in self.his_order:
                self.current_seekers.append(seeker)
                self.current_seekers_location .append(seeker.O_location)
        for seeker in self.remain_seekers:
            self.current_seekers.append(seeker)
            self.current_seekers_location .append(seeker.O_location)

        start = time.time()
        reward, done = self.process(self.time)
        end = time.time()
        # print('process 用时', end - start)
        return reward,  done

    #
    def process(self, time_, ):
        reward = 0
        takers = []
        vehicles = []
        seekers = self.current_seekers

        if self.time >= time.mktime(time.strptime(self.cfg.date + self.cfg.simulation_end_time, "%Y-%m-%d %H:%M:%S")):
            print('当前episode仿真时间结束,奖励为:', self.total_reward)

            # 计算系统指标
            self.res = {}
            # for seekers
            for order in self.his_order:
                self.waitingtime.append(order.waitingtime)
                self.traveltime.append(order.traveltime)
                self.ride_distance_error.append(
                    abs(order.rs - order.ride_distance))
                self.relative_ride_distance_error.append(
                    abs(order.rs - order.ride_distance) / order.rs)

            for order in self.carpool_order:
                self.detour_distance.append(order.detour)
                self.shared_distance_error.append(
                    abs(order.es - order.shared_distance))
                self.relative_shared_distance_error.append(
                    abs(order.es - order.shared_distance)/order.es)

            self.res['detour_distance'] = np.mean(self.detour_distance)
            self.res['waitingTime'] = np.mean(self.waitingtime)
            # for vehicle
            self.res['pickup_time'] = np.mean(self.pickup_time)
            self.res['shared_distance'] = np.mean(self.shared_distance)

            self.res['saved_ride_distance'] = np.sum(
                self.saved_travel_distance)


            self.res['platform_income'] = np.sum(self.platform_income)
            self.res['response_rate'] = len(
                list(set(self.his_order))) / len(self.order_list)
            self.res['carpool_rate'] = len(
                self.carpool_order) / len(self.his_order)
            self.res['ride_distance_error'] = self.ride_distance_error
            self.res['shared_distance_error'] = self.shared_distance_error
            self.res['relative_ride_distance_error'] = self.relative_ride_distance_error
            self.res['relative_shared_distance_error'] = self.relative_shared_distance_error

            # for system
            folder = 'output/'+self.cfg.date + '/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.save_metric(folder + 'system_metric.pkl')
            self.save_his_order(folder + 'history_order.pkl')
            return reward, True
        else:
            # print('当前episode仿真时间:',time_)
            # 判断智能体是否能执行动作
            self.activate_drivers = 0
            for vehicle in self.vehicle_list:
                # print('vehicle.activate_time',vehicle.activate_time)
                # print('vehicle.state',vehicle.state)
                vehicle.is_activate(time_)
                vehicle.reset_reposition()

                if vehicle.state == 1:  # 能执行动作
                    vehicles.append(vehicle)
            # print('len(vehicles)',len(vehicles),'len(takers)', len(takers))
            start = time.time()
            reward = self.batch_matching(vehicles, seekers)
            end = time.time()
            if self.cfg.progress_target:
                print('匹配用时{},time{},vehicles{},seekers{}'.format(
                    end - start, self.time_slot, len(vehicles), len(seekers)))
            # for vehicle in self.vehicle_list:
            #     print('vehicle.activate_time',vehicle.activate_time)
            #     print('vehicle.target',vehicle.target)
            self.total_reward += reward

            return reward,  False

    # 匹配算法
    def batch_matching(self, vehicles, seekers):
        import time
        start = time.time()
        # 构造权重矩阵
        demand = len(seekers)
        supply = len(vehicles)
        row_nums = demand + supply  # 加入乘客选择wait
        column_nums = demand + supply  # 加入司机选择wait
        # print('row_nums,column_nums ',row_nums,column_nums )
        dim = max(row_nums, column_nums)
        matrix = np.ones((dim, dim)) * -5000

        # 从乘客角度计算匹配权重
        for column in range(demand):
            # 当前seeker的zone
            location = seekers[column].O_location
            zone = self.network.Nodes[location].getZone()
            nodes = zone.nodes

            for row in range(supply):

                if vehicles[row].location in nodes:
                    start = time.time()
                    matrix[row, column] = self.calVehiclesWeights(vehicles[row], seekers[column],
                                                                  optimazition_target=self.optimazition_target,
                                                                  matching_condition=self.matching_condition)
                    end = time.time()
                    # print('计算Vehicle权重时间', end - start)
                else:
                    continue

        # 计算司机选择调度的权重
        for row in range(len(vehicles)):
            for column in range(len(seekers), dim):
                matrix[row, column] = - 3000

        # 计算乘客选择等待的权重
        for column in range(len(seekers)):
            for row in range(len(vehicles), row_nums):
                matrix[row, column] = -3000

        # 匹配
        if demand == 0 or supply == 0:
            self.remain_seekers = []
            for seeker in seekers:
                if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < self.cfg.delay_time_threshold:
                    seeker.set_delay(self.time)
                    self.remain_seekers.append(seeker)
            return 0

        import time
        start = time.time()
        matcher = KM_method.KM_method(matrix)
        res, weights = matcher.run()
        end = time.time()

        # print(res)

        for i in range(len(vehicles)):
            #  第i个vehicle响应第res[1][i]个订单
            if res[i] >= len(seekers):
                # 接到了虚拟订单
                vehicles[i].reposition_target = 1
                if not vehicles[i].path:
                    repostion_location = random.choice(self.locations)
                    distance, vehicles[i].path, vehicles[i].path_length = self.network.get_path(
                        vehicles[i].location, repostion_location)

            else:
                # print('vehicle id{},order id{}'.format(vehicles[i].id, seekers[res[i + len(takers)]].id))
                vehicles[i].order_list.append(seekers[res[i]])

                # 配对乘客
                if seekers[res[i]].id in self.matched_results: 
                    p1_id = self.matched_results[seekers[res[i]].id]
                    p1 = self.Seekers[p1_id]
                    vehicles[i].order_list.append(p1)
                    p1.response_target = 1
                    vehicles[i].target = 1 # 接到两个订单
                else:
                    vehicles[i].target = 0 # 接到一个订单
                seekers[res[i]].response_target = 1

        # 更新司機位置
        for vehicle in vehicles:
            if not vehicle.order_list:  # 沒接到乘客
                vehicle.reposition_target = 1
                # 更新空车司机位置
                if vehicle.path:
                    # 没到目的地
                    if self.time > vehicle.reposition_time + self.cfg.unit_driving_time * vehicle.path_length[0]:
                        vehicle.reposition_time = self.time
                        vehicle.location = vehicle.path.pop()
                        vehicle.path_length.pop()
                else:
                    # 到目的地了
                    vehicle.reposition_time = self.time

            elif vehicle.target == 1:
                # 依次接两个乘客
                pickup_p0_distance = self.get_path(
                    vehicle.location, vehicle.order_list[0].O_location)
                pickup_p1_distance = self.get_path(
                    vehicle.order_list[0].O_location, vehicle.order_list[1].O_location)
                self.pickup_time.append(self.cfg.unit_driving_time * pickup_p0_distance)
                self.pickup_time.append(self.cfg.unit_driving_time * pickup_p1_distance)
                fifo, distance = self.is_fifo(
                    vehicle.order_list[0], vehicle.order_list[1])
                if fifo:
                    shared_distance = self.get_path(
                        vehicle.order_list[1].O_location, vehicle.order_list[0].D_location)
                    p0_invehicle = pickup_p1_distance + distance[0]
                    p1_invehicle = sum(distance)
                    p0_detour = p0_invehicle - \
                        vehicle.order_list[0].shortest_distance
                    p1_detour = p1_invehicle - \
                        vehicle.order_list[1].shortest_distance
                    # 更新智能体的位置
                    vehicle.location = vehicle.order_list[1].D_location

                else:
                    shared_distance = vehicle.order_list[1].shortest_distance
                    p0_invehicle = pickup_p1_distance + sum(distance)
                    p1_invehicle = distance[0]
                    p0_detour = p0_invehicle - \
                        vehicle.order_list[0].shortest_distance
                    p1_detour = p1_invehicle - \
                        vehicle.order_list[1].shortest_distance
                    # 更新智能体的位置
                    vehicle.location = vehicle.order_list[0].D_location

                distance_saving = vehicle.order_list[1].shortest_distance + vehicle.order_list[0].shortest_distance - (
                    p0_invehicle + p1_invehicle - shared_distance)
                self.saved_travel_distance += distance_saving
                # 计算拼车距离
                self.shared_distance.append(distance[0])
                # 计算绕行距离
                self.detour_distance.append(p0_detour)
                self.detour_distance.append(p1_detour)
                # 计算收入
                self.platform_income.append(
                    self.cfg.discount_factor * (vehicle.order_list[0].value + vehicle.order_list[1].value) -
                    self.cfg.unit_distance_cost/1000 * (p0_invehicle + p1_invehicle -distance[0] + pickup_p0_distance)
                )
                # 更新智能体可以采取动作的时间
                vehicle.activate_time = self.time + self.cfg.unit_driving_time * \
                    (p0_invehicle + p1_invehicle -
                     shared_distance + pickup_p0_distance)
                self.his_order.append(vehicle.order_list[0])
                self.his_order.append(vehicle.order_list[1])
                self.carpool_order.append(vehicle.order_list[0])
                self.carpool_order.append(vehicle.order_list[1])

                # 完成订单
                vehicle.order_list = []
                vehicle.path = []
                vehicle.path_length = []
                vehicle.target = 0
            else:
                # 接到一个乘客
                pickup_distance = self.get_path(
                    vehicle.location, vehicle.order_list[0].O_location)
                distance = self.get_path(
                    vehicle.order_list[0].O_location, vehicle.order_list[0].O_location)
                self.his_order.append(vehicle.order_list[0])
                # 更新智能体可以采取动作的时间
                vehicle.activate_time = self.time + self.cfg.unit_driving_time * \
                    (pickup_distance + distance)
                self.pickup_time.append(self.cfg.unit_driving_time * pickup_distance)
                self.platform_income.append(
                        vehicle.order_list[0].value -
                        self.cfg.unit_distance_cost/1000 * ( vehicle.order_list[0].shortest_distance +pickup_distance)
                    )
                # 完成订单
                vehicle.order_list = []
                vehicle.path = []
                vehicle.path_length = []
                vehicle.target = 0

        self.remain_seekers = []
        for seeker in seekers:
            if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < seeker.waitingtime_threshold:
                seeker.set_delay(self.time)
                self.remain_seekers.append(seeker)

        return 0


    def calVehiclesWeights(self, vehicle, seeker,  optimazition_target, matching_condition):
        pick_up_distance = self.get_path(
            seeker.O_location, vehicle.location)
        return - pick_up_distance

    def is_fifo(self, p0, p1):
        fifo = [self.get_path(p1.O_location, p0.D_location),
                self.get_path(p0.D_location, p1.D_location)]
        lifo = [self.get_path(p1.O_location, p1.D_location),
                self.get_path(p1.D_location, p0.D_location)]
        if sum(fifo) < sum(lifo):
            return True, fifo
        else:
            return False, lifo

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


import pandas as pd
import numpy as np
import Network as net
import Seeker
import random
import Vehicle
import os
import Trip
from pulp import *
import time


class Simulation():

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.date = cfg.date
        self.order_list = pd.read_csv(self.cfg.order_file).sample(
            frac=cfg.demand_ratio, random_state=1)
        # print('order_list',len(self.order_list))
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

        self.time_unit = self.cfg.time_unit  # 控制的时间窗,每10s匹配一次
        self.index = 0  # 计数器
        self.device = cfg.device
        self.total_reward = 0
        self.optimazition_target = cfg.optimazition_target  # 仿真的优化目标
        self.matching_condition = cfg.matching_condition  # 匹配时是否有条件限制
        self.pickup_distance_threshold = cfg.pickup_distance_threshold
        self.detour_distance_threshold = cfg.detour_distance_threshold
        self.delay_time_threshold = cfg.delay_time_threshold
        
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
        time_old = self.time
        self.time += self.time_unit
        self.time_slot += 1

        # 筛选时间窗内的订单
        current_time_orders = self.order_list[self.order_list['beginTime_stamp'] > time_old]
        current_time_orders = current_time_orders[current_time_orders['beginTime_stamp'] <= self.time]
        self.current_seekers = []
        self.current_seekers_location = []
        for index, row in current_time_orders.iterrows():
            seeker = Seeker.Seeker( row)
            seeker.set_shortest_path(self.get_path(
                seeker.O_location, seeker.D_location))
            # 生成一个正态分布随机数
            # 设置随机数生成的种子
            np.random.seed(index)
            seeker.waitingtime_threshold = max(30, np.random.normal(self.cfg.delay_time_threshold, 10))
            # print('seeker.waitingtime_threshold',seeker.waitingtime_threshold)
            value = self.cfg.unit_distance_value / 1000 * seeker.shortest_distance
            seeker.set_value(value)
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

            # 计算系统指标
            self.res = {}
            # for seekers
            for order in self.his_order:
                self.waitingtime.append(order.waitingtime)

                self.traveltime.append(order.traveltime)

            for order in self.carpool_order:
                self.detour_distance.append(order.detour)
                # print('order.detour',order.detour)

            self.res['waitingTime'] = np.mean(self.waitingtime)
            self.res['traveltime'] = np.mean(self.traveltime)
            self.res['detour_distance'] = np.mean(self.detour_distance)

            # for vehicle
            self.res['pickup_time'] = np.mean(self.pickup_time)
            self.res['shared_distance'] = np.mean(self.shared_distance)
            self.res['total_ride_distance'] = np.sum(
                self.total_travel_distance)
            # self.res['saved_ride_distance'] = np.sum(
            #     self.saved_travel_distance) - np.sum(self.total_travel_distance)
            self.res['saved_ride_distance'] = np.sum(
                self.saved_travel_distance)
            self.res['platform_income'] = np.sum(self.platform_income)
            self.res['avg_platform_income']= np.sum(self.platform_income) / len(self.his_order)
            self.res['avg_platform_income']= np.mean(self.platform_income) 
            self.res['response_rate'] = len(
                list(set(self.his_order))) / len(self.order_list)
            self.res['carpool_rate'] = len(
                self.carpool_order) / len(self.his_order)

            # for system
            folder = 'output/'+self.cfg.date + '/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.visualization_df.to_csv(folder + 'visualization_df.csv')
            self.save_metric(folder + 'system_metric.pkl')
            self.save_his_order(folder + 'history_order.pkl')
            return reward, True
        else:
            # print('当前episode仿真时间:',time_)
            # 判断智能体是否能执行动作
            for vehicle in self.vehicle_list:
                # print('vehicle.activate_time',vehicle.activate_time)
                # print('vehicle.state',vehicle.state)
                vehicle.is_activate(time_)
                vehicle.reset_reposition()

                # if vehicle.target == 0 : # 能执行动作
                #     print('当前episode仿真时间:',time_)
                #     print('id{},vehicle.activate_time{}'.format(vehicle.id, vehicle.activate_time))
                #     print('激活时间{}'.format(vehicle.activate_time - time_))
                if vehicle.state == 1:  # 能执行动作
                    # print('id{},vehicle.target{}'.format(vehicle.id, vehicle.target))
                    if vehicle.target == 0:
                        vehicles.append(vehicle)
                    else:
                        # print('id{},vehicle.target{}'.format(vehicle.id, vehicle.target))
                        takers.append(vehicle)
            # print('len(vehicles)',len(vehicles),'len(takers)', len(takers))
            start = time.time()
                    # 匹配
            # if len(vehicles + takers) == 0 or len(seekers) == 0:
            #     self.remain_seekers = []
            #     for seeker in seekers:
            #         if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < self.delay_time_threshold:
            #             seeker.set_delay(self.time)
            #             self.remain_seekers.append(seeker)
            #     return 0, False
            rr, rv = self.generate_RV_graph(seekers, vehicles + takers)
            self.generate_RTV_graph(vehicles, takers, rr)
            reward = self.optimal_assignment(takers, vehicles, seekers)
            end = time.time()
            if self.cfg.progress_target:
                print('匹配用时{},time{},vehicles{},takers{},seekers{}'.format(end - start, self.time_slot, len(vehicles),len(takers) ,len(seekers)))   
            # for vehicle in self.vehicle_list:
            #     print('vehicle.activate_time',vehicle.activate_time)
            #     print('vehicle.target',vehicle.target)

            return reward,  False

    # 生成RV图
    def generate_RV_graph(self, seekers, vehicles):

        def calculate_rr_delays_innerloop(i, seekers, results):
            for j in range(i+1, len(seekers)):

                res = self.cal_rr_detour(seekers[j], seekers[i])
                if res:
                    results[res[0]] = res[1]
            return results

        def calculate_rr_delays_outerloop(seekers, results):

            for i in range(len(seekers)):
                results = calculate_rr_delays_innerloop(i, seekers, results)
                # print('len(i){},len(results){}'.format(i,len(results)))

            return results

        def calculate_rv_delays_innerloop(vehicle, seekers, results):
            for j in range(len(seekers)):

                res = self.cal_rv_detour(seekers[j], vehicle)
                if res:
                    results[res[0]] = res[1]
                    # print('res[0][1]',res[0][1])
                    vehicle.e_r_v.append(res[0][1])
            return results

        def calculate_rv_delays_outerloop(vehicles, seekers, results):
            for i in range(len(vehicles)):
                vehicles[i].e_r_v = []
                results = calculate_rv_delays_innerloop(
                    vehicles[i], seekers, results)
                # print('len(i){},len(results){}'.format(i,len(results)))
            return results

        results = {}
        e_r_r = calculate_rr_delays_outerloop(seekers, results)
        results = {}
        e_r_v = calculate_rv_delays_outerloop(vehicles, seekers, results)

        # print('len(e_r_r), len(e_r_v)',len(e_r_r), len(e_r_v))
        # if e_r_r:
        #     for key in e_r_r.keys():
        #         if key[0].id in self.his_order or key[1].id in self.his_order:
        #             print('error, p1 id{}, p2_id{}'.format(e_r_r[key][0].id,e_r_r[key][1].id ) )
        #             print(self.his_order)
        return e_r_r, e_r_v

    # 组建RTV图
    def generate_RTV_graph(self, vehicles, takers, e_r_r):

        # print(e_r_r.keys())

        # 给空车匹配
        for v in vehicles:
            v.trips = []
            # print('添加vehicle乘客的len(v.e_r_v)',len(v.e_r_v))
            # 添加一个乘客的trips
            trip_1 = []
            for r in v.e_r_v:
                # print('v.id{},r.id{}'.format(v.id,r.id))
                trip = Trip.Trip(r, False, v)
                # print('trip1生成')
                r.trips.append(trip)
                weight = self.calVehiclesWeights(v, r, self.cfg.optimazition_target, True)
                # print('weight',weight)
                trip.set_Value(weight)
                trip_1.append(trip)

            # 添加两个乘客的trips
            trip_2 = []
            for i in range(len(trip_1)):
                for j in range(i+1, len(trip_1)):
                    # print('(trip_1[i],trip_1[j])',(trip_1[i].r1,trip_1[j].r1))
                    # print('e_r_r.keys()',e_r_r.keys())
                    if (trip_1[i].r1, trip_1[j].r1) in e_r_r.keys() or (trip_1[j].r1, trip_1[i].r1) in e_r_r.keys():
                        if trip_1[i].r1 in v.e_r_v and trip_1[j].r1 in v.e_r_v:
                            trip = Trip.Trip(trip_1[i].r1, trip_1[j].r1, v)
                            trip_1[i].r1.trips.append(trip)
                            trip_1[j].r1.trips.append(trip)
                            if (trip_1[i].r1, trip_1[j].r1) in e_r_r.keys():
                                weight = e_r_r[(trip_1[i].r1, trip_1[j].r1)]
                            else:
                                weight = e_r_r[(trip_1[j].r1, trip_1[i].r1)]
                            # print('weight',weight)
                            trip.set_Value(weight)
                            trip_2.append(trip)

            v.trips = trip_1 + trip_2

        # 给taker匹配
        # print('e_r_r.keys()',e_r_r.keys())
        for v in takers:
            # 添加一个乘客的trips
            trips = []

            for r in v.e_r_v:
                # 判断这俩是否可以合乘
                res = self.cal_rr_detour(v.order_list[0], r)
                if res:
                    # trip = Trip.Trip(v.order_list[0] , r, v)
                    trip = Trip.Trip(r, False, v)
                    r.trips.append(trip)
                    weight = self.calTakersWeights(v, r, self.cfg.optimazition_target, True)
                    trip.set_Value(weight)
                    # print('weight',weight)
                    trips.append(trip)

            v.trips = trips

    # # 初始的贪婪分配
    # def greedy_assignment(self,takers, vacant_vehicles):
    #     Rok = []
    #     for taker in takers:
    #         for trip in taker.trips:

    #     return

    # 最优匹配算法

    def optimal_assignment(self, takers, vehicles, seekers):
        start = time.time()

        trips_num = 0
        trips_dic = {}
        for v in vehicles:
            for i in range(len(v.trips)):
                trips_dic[trips_num] = v.trips[i]
                # print('vehicles reward',trips_dic[trips_num].value)
                trips_num += 1
            v.trips_num = len(v.trips)

        for v in takers:
            for i in range(len(v.trips)):
                # print('v.trip.type',v.trips[i].target)
                trips_dic[trips_num] = v.trips[i]
                # print('taker reward',trips_dic[trips_num].value)
                trips_num += 1
            v.trips_num = len(v.trips)

        # 定义问题
        prob = LpProblem("optimal_assignment", LpMaximize)
        # print('trips',trips_num)
        # 定义0-1整数变量
        x = LpVariable.dicts("x", range(trips_num), cat=LpBinary)

        # 定义cost
        c = [trips_dic[i].value for i in range(trips_num)]

        # 添加目标函数
        prob += lpSum([c[i] * x[i] for i in range(trips_num)])

        # 添加约束条件
        # 每个司机最多响应一个trip
        lower_bound = 0
        for v in vehicles:
            upper_bound = lower_bound + v.trips_num
            prob += lpSum(x[i] for i in range(lower_bound, upper_bound)) <= 1
            lower_bound = upper_bound

        for taker in takers:
            upper_bound = lower_bound + taker.trips_num
            prob += lpSum(x[i] for i in range(lower_bound, upper_bound)) <= 1
            lower_bound = upper_bound

        # 每个乘客最多被一个司机响应
        for seeker in seekers:
            trips = seeker.trips
            tmp = []
            for trip in trips:
                for i in range(trips_num):
                    if trips_dic[i] == trip:
                        tmp.append(i)
            prob += lpSum(x[i] for i in tmp) <= 1

        # 求解问题
        prob.solve(PULP_CBC_CMD(msg=0))
        end = time.time()

        # print('求解优化问题用时:', end - start)
        # print('results')
        # for i in range(trips_num):
        #     print(x[i].varValue)

        # 输出结果
        matched_1_passenger_vehicle = []
        matched_2_passenger_vehicle = []

        for i in range(trips_num):
            if x[i].varValue == 1:
                # print(x[i])
                trip = trips_dic[i]
                # print('reward',trips_dic[i].value)
                if trip.target == 0:
                    # print('trip.r1.id',trip.r1.id)
                    # 加载到司机的执行订单列表
                    trip.v.order_list.append(trip.r1)
                    trip.v.reposition_target = 0
                    matched_1_passenger_vehicle.append(trip.v)
                    # print('普通订单',trip.v.id)
                    # # 记录系统指标
                    # print('time{},id{}'.format(self.time_slot, trip.r1.id))
                    # if trip.r1 in self.his_order:
                    #     print('接普通单 error!!!')
                    # self.his_order.append(trip.r1)
                    # # print('接到一个普通订单',trip.r1.id)
                    # if len(trip.v.order_list) == 2:
                    #     self.carpool_order.append(trip.r1)
                    #     self.carpool_order.append(trip.v.order_list[0])
                    # print('匹配到拼车订单，订单号',trip.v.order_list[0])
                    trip.r1.set_waitingtime(
                        self.time - trip.r1.begin_time_stamp)
                    trip.r1.response_target = 1

                else:
                    # print('trip.r1.id',trip.r1.id,'trip.r2.id',trip.r2.id)
                    # 加载到司机的执行订单列表
                    trip.v.order_list.append(trip.r1)
                    trip.v.order_list.append(trip.r2)

                    # print('time{},r1_id{},r2_id{},v.type{},orderlist{}'.format(self.time_slot, trip.r1.id, trip.r2.id,trip.v.target,len(trip.v.order_list)))
                    # if trip.r1 in self.his_order:
                    #     print('trip.r1 error!!!')
                    # if trip.r2 in self.his_order:
                    #     print('trip.r2 error!!!')

                    # print('拼车订单',trip.v.id)
                    matched_2_passenger_vehicle.append(trip.v)
                    trip.v.reposition_target = 0

                    trip.r1.set_waitingtime(
                        self.time - trip.r1.begin_time_stamp)
                    trip.r1.response_target = 1

                    trip.r2.set_waitingtime(
                        self.time - trip.r2.begin_time_stamp)
                    trip.r2.response_target = 1

        # unmatched passengers
        self.remain_seekers = []
        for seeker in seekers:
            # if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < self.delay_time_threshold * seeker.random_seed:
            if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < seeker.waitingtime_threshold:
                seeker.set_delay(self.time)
                self.remain_seekers.append(seeker)

        # 更新位置
        start = time.time()
        for taker in takers:
            # 先判断taker 接没接到乘客
            # 当前匹配没拼到新乘客
            if taker not in matched_1_passenger_vehicle:
                # 是否超出匹配时间
                if self.time - taker.order_list[0].begin_time_stamp - \
                        self.cfg.unit_driving_time * taker.p0_pickup_distance > 600:
                    # 已经10分钟还没拼到车，派送到终点
                    # 派送时间
                    travel_distance = self.get_path(
                        taker.location, taker.order_list[0].D_location)

                    taker.drive_distance += travel_distance
                    taker.order_list[0].ride_distance = self.get_path(
                        taker.order_list[0].O_location, taker.order_list[0].D_location)
                    taker.order_list[0].shared_distance = 0
                    self.total_travel_distance += taker.p0_pickup_distance
                    self.total_travel_distance += taker.order_list[0].ride_distance
                    # self.saved_travel_distance += taker.order_list[0].ride_distance

                    self.platform_income.append(
                       self.cfg.discount_factor *  taker.order_list[0].value -
                        self.cfg.unit_distance_cost/1000 * ( taker.order_list[0].shortest_distance + taker.p0_pickup_distance)
                    )
                    # print('order.value{},driver_distance{},income{}:'.format(taker.order_list[0].value,taker.drive_distance,self.platform_income[-1]))
                    # 更新智能体可以采取动作的时间
                    taker.activate_time += self.cfg.unit_driving_time * travel_distance
                    self.his_order.append(taker.order_list[0])
                    # print('没有拼到车，activate_time:{}'.format(taker.activate_time - self.time))
                    # 更新智能体的位置
                    taker.location = taker.order_list[0].D_location
                    # 完成订单
                    taker.order_list = []
                    taker.target = 0  # 变成vehicle
                    taker.drive_distance = 0
                    taker.reward = 0
                    taker.p0_pickup_distance = 0
                    taker.path = []
                    taker.path_length = []

                # 没超出匹配时间
                # 判断是否接到第一个乘客
                elif taker.origin_location != taker.order_list[0].O_location:
                    pickup_distance, taker.path, taker.path_length = self.network.get_path(
                        taker.location, taker.order_list[0].O_location)
                    if taker.path:
                        # 还没到下一个节点
                        if self.time - taker.activate_time > self.cfg.unit_driving_time * taker.path_length[0]:
                            taker.location = taker.path.pop()
                            pickup_distance = taker.path_length.pop()
                            taker.activate_time = self.time
                            taker.p0_pickup_distance += pickup_distance
                            taker.drive_distance += pickup_distance

                    else:
                        # 接到乘客了
                        pickup_time = self.cfg.unit_driving_time * taker.p0_pickup_distance
                        self.pickup_time.append(pickup_time)
                        taker.location = taker.order_list[0].O_location
                        taker.activate_time = self.time
                        taker.origin_location = taker.order_list[0].O_location
                
                # 没匹配到其他乘客，但接到p1了，开始在路上派送
                else:
                    distance, taker.path, taker.path_length = self.network.get_path(
                        taker.location, taker.order_list[0].D_location)
                    # 没送到目的地
                    if taker.path:
                        if self.time - taker.activate_time > self.cfg.unit_driving_time * taker.path_length[0]:
                            taker.location = taker.path.pop()
                            distance = taker.path_length.pop()
                            taker.drive_distance += distance

                    else:
                        # 送到目的地了

                        taker.location = taker.order_list[0].D_location
                        taker.origin_location = taker.order_list[0].D_location
                        taker.activate_time = self.time
                        # 乘客的行驶距离
                        taker.order_list[0].ride_distance = taker.drive_distance
                        taker.order_list[0].shared_distance = 0
                        self.total_travel_distance += taker.p0_pickup_distance
                        self.total_travel_distance += taker.order_list[0].ride_distance
                        # self.saved_travel_distance += taker.order_list[0].ride_distance
                        # 计算平台收益
                        self.platform_income.append(self.cfg.discount_factor * taker.order_list[0].value -
                                                    self.cfg.unit_distance_cost/1000 * ( taker.order_list[0].shortest_distance + taker.p0_pickup_distance)
                                                    )

                        # 完成订单
                        self.his_order.append(taker.order_list[0])
                        taker.order_list = []
                        taker.target = 0  # 变成vehicle
                        taker.drive_distance = 0
                        taker.reward = 0
                        taker.p0_pickup_distance = 0
                        taker.path = []
                        taker.path_length = []

            # 当前匹配有新乘客
            else:
                # 判断p0是否已经上车，没上车的话先接p0
                if taker.origin_location != taker.order_list[0].O_location:
                    pickup_distance = self.get_path(
                        taker.location, taker.order_list[0].O_location)
                    taker.origin_location = taker.order_list[0].O_location
                    taker.location = taker.order_list[0].O_location
                    taker.p0_pickup_distance += pickup_distance
                    pickup_time = self.cfg.unit_driving_time * taker.p0_pickup_distance
                    taker.drive_distance += pickup_distance
                    self.pickup_time.append(pickup_time)

                # 接新乘客
                pickup_distance, taker.path, taker.path_length \
                    = self.network.get_path(
                        taker.order_list[0].O_location, taker.order_list[1].O_location)

                pickup_time = self.cfg.unit_driving_time * pickup_distance
                # print('pickup_time',pickup_time)

                taker.drive_distance += pickup_distance
                self.pickup_time.append(pickup_time)
                taker.p1_pickup_distance = pickup_distance
                taker.order_list[0].ride_distance += pickup_distance

                self.total_travel_distance += taker.p0_pickup_distance
                self.total_travel_distance += taker.p1_pickup_distance

                # self.saved_travel_distance += taker.order_list[0].shortest_distance
                # self.saved_travel_distance += taker.order_list[1].shortest_distance
                
                self.his_order.append(taker.order_list[0])
                self.his_order.append(taker.order_list[1])

                self.carpool_order.append(taker.order_list[0])
                self.carpool_order.append(taker.order_list[1])

                # 决定派送顺序，是否fifo
                fifo, distance = self.is_fifo(
                    taker.order_list[0], taker.order_list[1])
                if fifo:
                    # 先上先下
                    self.total_travel_distance += self.get_path(
                        taker.order_list[1].O_location, taker.order_list[0].D_location)
                    self.total_travel_distance += self.get_path(
                        taker.order_list[0].D_location, taker.order_list[1].D_location)

                    p0_invehicle = taker.p1_pickup_distance  + distance[0]
                    p0_expected_distance = taker.order_list[0].shortest_distance
                    # 绕行
                    taker.order_list[0].set_detour(
                        p0_invehicle - p0_expected_distance)
                    # print('detour',p0_invehicle - p0_expected_distance)
                    p1_invehicle = sum(distance)
                    p1_expected_distance = taker.order_list[1].shortest_distance
                    taker.order_list[1].set_detour(
                        p1_invehicle - p1_expected_distance)
                    # print('detour',p1_invehicle - p1_expected_distance)
                    # travel time
                    taker.order_list[0].set_traveltime(
                        self.cfg.unit_driving_time * p0_invehicle)
                    taker.order_list[1].set_traveltime(
                        self.cfg.unit_driving_time * p1_invehicle)

                else:
                    # 先上后下
                    self.total_travel_distance += self.get_path(
                        taker.order_list[1].O_location, taker.order_list[1].D_location)
                    self.total_travel_distance += self.get_path(
                        taker.order_list[1].D_location, taker.order_list[0].D_location)

                    p0_invehicle = taker.p1_pickup_distance + sum(distance)
                    p1_invehicle = distance[0]
                    p0_expected_distance = taker.order_list[0].shortest_distance
                    p1_expected_distance = taker.order_list[1].shortest_distance
                    taker.order_list[0].set_detour(
                        p0_invehicle - p0_expected_distance)
                    # print('detour',p0_invehicle - p0_expected_distance)
                    taker.order_list[1].set_detour(
                        p1_invehicle - taker.order_list[1].shortest_distance)
                    # print('detour',p1_invehicle - p1_expected_distance)
                    # travel time
                    taker.order_list[0].set_traveltime(
                        self.cfg.unit_driving_time * p0_invehicle)
                    taker.order_list[1].set_traveltime(
                        self.cfg.unit_driving_time * p1_invehicle)

                # 乘客的行驶距离
                taker.order_list[0].ride_distance = p0_invehicle
                taker.order_list[1].ride_distance = p1_invehicle
                taker.order_list[0].shared_distance = distance[0]
                taker.order_list[1].shared_distance = distance[0]
                self.saved_travel_distance += taker.order_list[1].shortest_distance + taker.order_list[0].shortest_distance - (p0_invehicle + p1_invehicle - distance[0]) 
                taker.drive_distance += sum(distance)

                # 计算平台收益
                self.platform_income.append(
                    self.cfg.discount_factor * (taker.order_list[0].value + taker.order_list[1].value) -
                    self.cfg.unit_distance_cost/1000 * (p0_invehicle + p1_invehicle -distance[0] + taker.p0_pickup_distance)
                )
                # print('order.value{},driver_distance{},income:{}'.format(self.cfg.discount_factor * (taker.order_list[0].value + taker.order_list[1].value),taker.drive_distance,\
                #     self.cfg.discount_factor * (taker.order_list[0].value + taker.order_list[1].value) - \
                #     self.cfg.unit_distance_cost/1000 * taker.drive_distance))
                # 计算拼车距离
                self.shared_distance.append(distance[0])
                # 更新智能体可以采取动作的时间
                # 计算司机完成两个订单需要的时间
                dispatching_time = pickup_time + \
                    self.cfg.unit_driving_time * sum(distance)
                taker.activate_time = self.time + dispatching_time
                # print('拼车完成，activate_time:{}'.format(taker.activate_time - self.time))
                # 更新智能体的位置
                taker.location = taker.order_list[1].D_location
                taker.origin_location = taker.order_list[1].D_location
                # 完成订单
                taker.order_list = []
                taker.target = 0  # 变成vehicle
                taker.drive_distance = 0
                taker.reward = 0
                taker.p0_pickup_distance = 0
                taker.p1_pickup_distance = 0
                taker.path = []
                taker.path_length = []

        for vehicle in vehicles:
            if vehicle not in matched_1_passenger_vehicle and vehicle not in matched_2_passenger_vehicle:
                # print('vehicle.id{},vehicle.reposition_target{}'.format(vehicle.id, vehicle.reposition_target))
                # 随机调度
                repostion_location = random.choice(self.locations)
                distance, vehicle.path, vehicle.path_length = self.network.get_path(
                        vehicle.location, repostion_location)
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

            elif vehicle in matched_1_passenger_vehicle:
                vehicle.target = 1  # 变成taker
                vehicle.origin_location = vehicle.location
                # print('vehicle.id{},vehicle.order_list{}'.format(vehicle.id,len(vehicle.order_list)))
                # print(vehicle.reposition_target)
                # 接新乘客
                pickup_distance, vehicle.path, vehicle.path_length = self.network.get_path(
                    vehicle.location, vehicle.order_list[0].O_location)

                if vehicle.path:
                    if self.time - vehicle.activate_time > self.cfg.unit_driving_time * vehicle.path_length[0]:
                        vehicle.location = vehicle.path.pop()
                        pickup_distance = vehicle.path_length.pop()
                        vehicle.activate_time = self.time
                        vehicle.p0_pickup_distance += pickup_distance
                        vehicle.drive_distance += pickup_distance

            else:
                # 一次匹配到两个乘客

                if vehicle.order_list[0].begin_time_stamp <= vehicle.order_list[1].begin_time_stamp:
                    p0 = vehicle.order_list[0]
                    p1 = vehicle.order_list[1]
                    pickup_distance = self.get_path(
                    vehicle.location, p0.O_location)
                    pickup_p1_time = self.cfg.unit_driving_time * pickup_distance
                    self.pickup_time.append(pickup_p1_time)
                    pickup_p2_time = self.cfg.unit_driving_time * self.get_path(
                        vehicle.order_list[0].O_location, vehicle.order_list[1].O_location)
                    self.pickup_time.append(pickup_p2_time)
                    # print('pickup_p1_time{},pickup_p2_time{}'.format(pickup_p1_time,pickup_p2_time))
                    # print('p1_id{},p2_id{}'.format(vehicle.order_list[0].id,vehicle.order_list[1].id))

                    vehicle.drive_distance += pickup_distance
                    vehicle.drive_distance += pickup_p2_time / self.cfg.unit_driving_time
                    self.total_travel_distance += pickup_distance
                    self.total_travel_distance += pickup_p2_time / self.cfg.unit_driving_time
                    vehicle.p0_pickup_distance = pickup_distance

                else:
                    p0 = vehicle.order_list[1]
                    p1 = vehicle.order_list[0]
                    pickup_distance = self.get_path(
                    vehicle.location, p0.O_location)
                    pickup_p2_time = self.cfg.unit_driving_time * pickup_distance
                    self.pickup_time.append(pickup_p2_time)
                    pickup_p1_time = self.cfg.unit_driving_time * self.get_path(
                        vehicle.order_list[1].O_location, vehicle.order_list[0].O_location)
                    self.pickup_time.append(pickup_p1_time)
                    # print('pickup_p1_time{},pickup_p2_time{}'.format(pickup_p1_time,pickup_p2_time))
                    # print('p1_id{},p2_id{}'.format(vehicle.order_list[0].id,vehicle.order_list[1].id))

                    vehicle.drive_distance += pickup_distance
                    vehicle.drive_distance += pickup_p1_time / self.cfg.unit_driving_time
                    self.total_travel_distance += pickup_distance
                    self.total_travel_distance += pickup_p1_time / self.cfg.unit_driving_time
                    vehicle.p0_pickup_distance = pickup_distance

                self.his_order.append(vehicle.order_list[0])
                self.his_order.append(vehicle.order_list[1])

                self.carpool_order.append(vehicle.order_list[0])
                self.carpool_order.append(vehicle.order_list[1])

                # 决定派送顺序，是否fifo
                fifo, distance = self.is_fifo(p0,p1)
                pickup_time = self.cfg.unit_driving_time * pickup_distance
                if fifo:
                    # 先上先下
                    self.total_travel_distance += self.get_path(
                        p1.O_location, p0.D_location)
                    self.total_travel_distance += self.get_path(
                        p0.D_location, p1.D_location)

                    p0_invehicle = pickup_distance + distance[0]
                    p0_expected_distance = p0.shortest_distance
                    # 绕行
                    p0.set_detour(
                        p0_invehicle - p0_expected_distance)
                    # print('detour',p0_invehicle - p0_expected_distance)
                    p1_invehicle = sum(distance)
                    p1_expected_distance = p1.shortest_distance
                    p1.set_detour(
                        p1_invehicle - p1_expected_distance)
                    # print('detour',p1_invehicle - p1_expected_distance)
                    # travel time
                    p0.set_traveltime(
                        self.cfg.unit_driving_time * p0_invehicle)
                    p1.set_traveltime(
                        self.cfg.unit_driving_time * p1_invehicle)
                    destination = p1.D_location
                else:
                    # 先上后下
                    self.total_travel_distance += self.get_path(
                        p1.O_location, p1.D_location)
                    self.total_travel_distance += self.get_path(
                        p1.D_location, p0.D_location)

                    p0_invehicle = pickup_distance + sum(distance)
                    p1_invehicle = distance[0]
                    p0_expected_distance = p0.shortest_distance
                    p0.set_detour(
                        p0_invehicle - p0_expected_distance)
                    # print('detour',p0_invehicle - p0_expected_distance)
                    p1_expected_distance = p1.shortest_distance
                    p1.set_detour(
                        p1_invehicle - p1.shortest_distance)
                    # print('detour',p1_invehicle - p1_expected_distance)
                    # travel time
                    p0.set_traveltime(
                        self.cfg.unit_driving_time * p0_invehicle)
                    p1.set_traveltime(
                        self.cfg.unit_driving_time * p1_invehicle)
                    destination = p0.D_location

                # 乘客的行驶距离
                p0.ride_distance = p0_invehicle
                p1.ride_distance = p1_invehicle
                p0.shared_distance = distance[0]
                p1.shared_distance = distance[0]
                self.saved_travel_distance += vehicle.order_list[1].shortest_distance + vehicle.order_list[0].shortest_distance - (p0_invehicle + p1_invehicle - distance[0]) 
                vehicle.drive_distance += sum(distance)

                # 计算平台收益
                self.platform_income.append(
                    self.cfg.discount_factor * (vehicle.order_list[0].value + vehicle.order_list[1].value) -
                    self.cfg.unit_distance_cost/1000 * (p0_invehicle + p1_invehicle -distance[0] + pickup_distance)
                )
                # print('order.value{},driver_distance{},income:{}'.format(self.cfg.discount_factor * (taker.order_list[0].value + taker.order_list[1].value),taker.drive_distance,\
                #     self.cfg.discount_factor * (taker.order_list[0].value + taker.order_list[1].value) - \
                #     self.cfg.unit_distance_cost/1000 * taker.drive_distance))
                # 计算拼车距离
                self.shared_distance.append(distance[0])
                # 更新智能体可以采取动作的时间
                # 计算司机完成两个订单需要的时间
                dispatching_time = pickup_time + \
                    self.cfg.unit_driving_time * sum(distance)
                vehicle.activate_time = self.time + dispatching_time
                # print('拼车完成，activate_time:{}'.format(taker.activate_time - self.time))
                # 更新智能体的位置
                vehicle.location = destination
                vehicle.origin_location = destination
                # 完成订单
                vehicle.order_list = []
                vehicle.target = 0  # 变成vehicle
                vehicle.drive_distance = 0
                vehicle.reward = 0
                vehicle.p0_pickup_distance = 0
                vehicle.p1_pickup_distance = 0
                vehicle.path = []
                vehicle.path_length = []

        end = time.time()
        # print('派送用时{},takers{},vehicles{}'.format(end-start, len(takers), len(vehicles)))

        return

    def calTakersWeights(self, taker, seeker,  optimazition_target, matching_condition):
        # expected shared distance
        pick_up_distance = self.get_path(
             taker.order_list[0].O_location,seeker.O_location)
        fifo, distance = self.is_fifo(taker.order_list[0], seeker)

        if fifo:
            shared_distance = self.get_path(
                seeker.O_location, taker.order_list[0].D_location)
            p0_invehicle = pick_up_distance + distance[0]
            p1_invehicle = sum(distance)
            p0_detour = p0_invehicle - taker.order_list[0].shortest_distance
            p1_detour = p1_invehicle - seeker.shortest_distance
            

        else:
            shared_distance = seeker.shortest_distance
            p0_invehicle = pick_up_distance + sum(distance)
            p1_invehicle = distance[0]
            p0_detour = p0_invehicle - taker.order_list[0].shortest_distance
            p1_detour = p1_invehicle - seeker.shortest_distance

        distance_saving = seeker.shortest_distance + taker.order_list[0].shortest_distance - (p0_invehicle + p1_invehicle - shared_distance)

        if optimazition_target == 'platform_income':
            if matching_condition and (pick_up_distance > self.pickup_distance_threshold or
                                    p0_detour > self.detour_distance_threshold or
                                    p1_detour > self.detour_distance_threshold):
                # print('detour_distance not pass', detour_distance)
                return self.cfg.dead_value
            else:
                profit = self.cfg.discount_factor * self.cfg.unit_distance_value / 1000 * (seeker.shortest_distance + taker.order_list[0].shortest_distance) - \
                self.cfg.unit_distance_cost/1000 *(p0_invehicle + p1_invehicle - shared_distance +  taker.p0_pickup_distance)
                return profit
            
        
        elif optimazition_target == 'combination':  
            profit = self.cfg.discount_factor * self.cfg.unit_distance_value / 1000 * (seeker.shortest_distance + taker.order_list[0].shortest_distance) - \
                self.cfg.unit_distance_cost/1000 *(p0_invehicle + p1_invehicle - shared_distance +  pick_up_distance)
            
            return self.cfg.beta * distance_saving + (1-self.cfg.beta) * profit 
        else:
            if matching_condition and (pick_up_distance > self.pickup_distance_threshold or
                                    p0_detour > self.detour_distance_threshold or
                                    p1_detour > self.detour_distance_threshold):
                # print('detour_distance not pass', detour_distance)
                return self.cfg.dead_value
            else:
                reward = (seeker.shortest_distance + taker.order_list[0].shortest_distance
                      - (p0_invehicle + p1_invehicle - shared_distance) - pick_up_distance) + 3000
                # print('taker reward',reward)
                return reward


    def calVehiclesWeights(self, vehicle, seeker,  optimazition_target, matching_condition):
        pick_up_distance = self.get_path(
            seeker.O_location, vehicle.location)
        
        if optimazition_target == 'platform_income':
            if matching_condition and (pick_up_distance > self.cfg.pickup_distance_threshold):
                return self.cfg.dead_value
            else:
                profit = self.cfg.discount_factor * self.cfg.unit_distance_value / 1000 * seeker.shortest_distance - \
                self.cfg.unit_distance_cost/1000 *(pick_up_distance + seeker.shortest_distance )
                return profit
            
        
        elif optimazition_target == 'combination':
            distance_saving =  (0- pick_up_distance) + 3000
            profit = self.cfg.unit_distance_value / 1000 * seeker.shortest_distance - \
                self.cfg.unit_distance_cost/1000 *(pick_up_distance)
            return self.cfg.beta * distance_saving + (1-self.cfg.beta) * profit 
        else:
            if matching_condition and (pick_up_distance > self.cfg.pickup_distance_threshold):
                return self.cfg.dead_value
            else:
                reward = (0- pick_up_distance) + 3000
                # print('vehicle reward',reward)
                return reward

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
    def cal_rr_detour(self, p0, p1):
        pick_up_distance = self.get_path(
            p0.O_location, p1.O_location)

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
        if p0_detour < self.cfg.detour_distance_threshold and p1_detour < self.cfg.detour_distance_threshold:
            # return [(p0, p1), shared_distance]
            if self.cfg.optimazition_target == 'expected_shared_distance':
                return [(p0, p1), (p0.shortest_distance + p1.shortest_distance) -(p0_invehicle + p1_invehicle -shared_distance)]
            else:
                reward = self.cfg.discount_factor * self.cfg.unit_distance_value / 1000 * (p0.shortest_distance + p1.shortest_distance) - \
                self.cfg.unit_distance_cost/1000 *(p0_invehicle + p1_invehicle - shared_distance +  pick_up_distance)
                return[(p0, p1), reward]
        else:
            return False

    # 判断某个乘客和某个司机是否可以组合成e(r,v)
    def cal_rv_detour(self, p, v):

        pick_up_distance = self.get_path(
            v.location, p.O_location)
        if pick_up_distance < self.cfg.pickup_distance_threshold:

            if len(v.order_list) == 0:
                # vacant vehicle
                # return [(v, p), p.es]
                return [(v, p), 3000 - pick_up_distance]
            else:
                # partially occupied vehicle

                fifo, distance = self.is_fifo(v.order_list[0], p)
                if fifo:
                    shared_distance = self.get_path(
                        p.O_location, v.order_list[0].D_location)
                    p0_invehicle = pick_up_distance + distance[0]
                    p1_invehicle = sum(distance)
                    p0_detour = p0_invehicle - \
                        v.order_list[0].shortest_distance
                    p1_detour = p1_invehicle - p.shortest_distance

                else:
                    shared_distance = p.shortest_distance
                    p0_invehicle = pick_up_distance + sum(distance)
                    p1_invehicle = distance[0]
                    p0_detour = p0_invehicle - \
                        v.order_list[0].shortest_distance
                    p1_detour = p1_invehicle - p.shortest_distance
                if p0_detour < self.cfg.detour_distance_threshold and p1_detour < self.cfg.detour_distance_threshold:
                    if self.cfg.optimazition_target == 'expected_shared_distance':
                        return [(v, p), (v.order_list[0].shortest_distance + p.shortest_distance)  - (p0_invehicle + p1_invehicle -shared_distance)]
                    else:
                        reward = self.cfg.discount_factor * self.cfg.unit_distance_value / 1000 * (v.order_list[0].shortest_distance + p.shortest_distance) - \
                        self.cfg.unit_distance_cost/1000 *(p0_invehicle + p1_invehicle - shared_distance +  pick_up_distance)
                        return[(v, p), reward]
                else:
                    return False
        else:
            return False

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

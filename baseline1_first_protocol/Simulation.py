
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
        self.order_list = pd.read_csv(self.cfg.order_file).sample(frac=cfg.demand_ratio, random_state=1)
        self.vehicle_num = int(len(self.order_list) / cfg.order_driver_ratio )
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
        self.visualization_df = pd.DataFrame(columns= ['vehicle_id','vehicle_type','depart_time','arrival_time','O_location','D_location','p1_id','p2_id'] )
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
        self.order_list = pd.read_csv(self.cfg.order_file).sample(frac=self.cfg.demand_ratio, random_state=1)
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
        self.visualization_df = pd.DataFrame(columns= ['vehicle_id','vehicle_type','depart_time','arrival_time','O_location','D_location','p1_id','p2_id'] )

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
        current_time_orders = self.order_list[self.order_list['beginTime_stamp'] >time_old]
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
            seeker.waitingtime_threshold = np.random.normal(self.cfg.delay_time_threshold, 10)
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
            print('当前episode仿真时间结束,奖励为:', self.total_reward)

            # 计算系统指标
            self.res = {}
            # for seekers
            for order in self.his_order:
                self.waitingtime.append(order.waitingtime)
                self.traveltime.append(order.traveltime)
            for order in self.carpool_order:
                self.detour_distance.append(order.detour)
            self.res['waitingTime'] = np.mean(self.waitingtime)
            self.res['traveltime'] = np.mean(self.traveltime)
            self.res['detour_distance'] = np.mean(self.detour_distance)

            # for vehicle
            self.res['pickup_time'] = np.mean(self.pickup_time)
            self.res['shared_distance'] = np.mean(self.shared_distance)
            self.res['total_ride_distance'] = np.sum(self.total_travel_distance)
            # self.res['saved_ride_distance'] = np.sum(self.saved_travel_distance) - np.sum(self.total_travel_distance)
            self.res['saved_ride_distance'] = np.sum(
                self.saved_travel_distance)

            self.res['platform_income'] = np.sum(self.platform_income)
            self.res['response_rate'] = len(list(set(self.his_order))) / len(self.order_list)
            self.res['carpool_rate'] = len(self.carpool_order) / len(self.his_order)

            # for system
            folder = 'output/'+self.cfg.date +'/'
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
                vehicle.is_activate(time_)
                vehicle.reset_reposition()

                if vehicle.state == 1:  # 能执行动作
                    # print('id{},vehicle.target{}'.format(vehicle.id, vehicle.target))
                    if vehicle.target == 0:
                        vehicles.append(vehicle)
                    else:
                        # print('id{},vehicle.target{}'.format(vehicle.id, vehicle.target))
                        takers.append(vehicle)
            # print('len(vehicles)',len(vehicles),'len(takers)', len(takers))
            start = time.time()
            reward = self.first_protocol_matching(takers, vehicles, seekers)
            end = time.time()
            if self.cfg.progress_target:
                print('匹配用时{},time{},vehicles{},takers{},seekers{}'.format(end - start, self.time_slot, len(vehicles),len(takers) ,len(seekers)))   
            # for vehicle in self.vehicle_list:
            #     print('vehicle.activate_time',vehicle.activate_time)
            #     print('vehicle.target',vehicle.target)
            self.total_reward += reward

            return reward,  False

    # 匹配算法
    def first_protocol_matching(self, takers, vehicles, seekers):
        if len(seekers) == 0:
            return 0
        import time
        start = time.time()


        for seeker in seekers:
            # 当前seeker的zone
            location = seeker.O_location
            zone = self.network.Nodes[location].getZone()
            nodes = zone.nodes
            match = {}
            for taker in takers:
                if taker.location in nodes:
                    match[taker] = self.calTakersWeights(seeker, taker, optimazition_target = self.optimazition_target, \
                matching_condition = self.matching_condition)

            for vehicle in vehicles:
                if vehicle.location in nodes:
                    match[vehicle] = self.calVehiclesWeights(vehicle,  seeker, optimazition_target = self.optimazition_target, \
                matching_condition = self.matching_condition)

            # 計算匹配結果
            if not match.values():
                seeker.response_taget = 0
            elif max(match.values()) == self.cfg.dead_value:
                seeker.response_taget = 0
            else:
                weight = max(match.values())
                seeker.response_taget = 1
                for key in match.keys():
                    if match[key] == weight: # key 接到乘客
                        if key.target == 1: # taker
                            key.order_list.append(seeker)
                            takers.remove(key)
                            # 记录目的地
                            pickup_distance, key.path, key.path_length = self.network.get_path(
                                key.location, key.order_list[1].O_location)

                        else: # vacant vehicle
                            key.order_list.append(seeker)
                            vehicles.remove(key)
                            # 更新目的地
                            pickup_distance, key.path, key.path_length \
                            = self.network.get_path(
                                key.location, key.order_list[0].O_location)


                        # 记录seeker等待时间
                        seeker.set_waitingtime(
                                self.time - seeker.begin_time_stamp)
                        seeker.response_target = 1   
                        break                     

        # 更新司機位置
        for vehicle in vehicles:
            if not vehicle.order_list: # 沒接到乘客
                vehicle.reposition_target = 1
                if not vehicle.path:
                    repostion_location = random.choice(self.locations)
                    distance, vehicle.path, vehicle.path_length = self.network.get_path(
                            vehicle.location, repostion_location)

        for taker in takers:
            if len(taker.order_list) != 2: # 沒接到乘客
                taker.reposition_target = 1

        # 更新位置
        for taker in takers:
            # 先判断taker 接没接到乘客
            # 当前匹配没拼到新乘客
            if taker.reposition_target == 1:
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
                        self.cfg.discount_factor * taker.order_list[0].value -
                        self.cfg.unit_distance_cost/1000 * taker.drive_distance
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
                            taker.path_length.pop()
                            taker.activate_time = self.time
                            taker.p0_pickup_distance += pickup_distance
                            taker.drive_distance += pickup_distance

                    else:
                        # 接到乘客了
                        taker.p0_pickup_distance += pickup_distance
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
                            taker.path_length.pop()
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
                                                    self.cfg.unit_distance_cost/1000 * taker.drive_distance
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
                    self.cfg.unit_distance_cost/1000 * taker.drive_distance
                )

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
            if vehicle.reposition_target == 1:
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

            else:
                vehicle.target = 1  # 变成taker
                vehicle.origin_location = vehicle.location
                # 接新乘客
                pickup_distance, vehicle.path, vehicle.path_length = self.network.get_path(
                    vehicle.location, vehicle.order_list[0].O_location)


        end = time.time()
        # print('派送用时{},takers{},vehicles{}'.format(end-start, len(takers), len(vehicles)))

        self.remain_seekers = []
        for seeker in seekers:
            # if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < self.cfg.delay_time_threshold * seeker.random_seed:
            if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < seeker.waitingtime_threshold:
                seeker.set_delay(self.time)
                self.remain_seekers.append(seeker)

        return 0



    def calTakersWeights(self,  seeker, taker,  optimazition_target, matching_condition):
        # expected shared distance
        pick_up_distance = self.get_path(
            seeker.O_location, taker.order_list[0].O_location)
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

        if matching_condition and (pick_up_distance > self.pickup_distance_threshold or
                                   p0_detour > self.detour_distance_threshold or
                                   p1_detour > self.detour_distance_threshold):
            # print('detour_distance not pass', detour_distance)
            return self.cfg.dead_value
        else:
            reward = (seeker.shortest_distance + taker.order_list[0].shortest_distance
                      - (p0_invehicle + p1_invehicle - shared_distance) - pick_up_distance) 
            # reward = ( seeker.shortest_distance + taker.order_list[0].shortest_distance - (p0_invehicle + p1_invehicle - shared_distance) )

            return reward



    def calVehiclesWeights(self, vehicle, seeker,  optimazition_target, matching_condition):

        pick_up_distance = self.get_path(
            seeker.O_location, vehicle.location)
        if matching_condition and (pick_up_distance > self.cfg.pickup_distance_threshold):
            return self.cfg.dead_value
        else:
            reward = (0 - pick_up_distance)
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


    def get_path(self, O, D):
        tmp = self.shortest_path[(self.shortest_path['O'] == O) & (
            self.shortest_path['D'] == D)]
        if tmp['distance'].unique():
            return tmp['distance'].unique()[0]
        else:
            return self.network.get_path(O,D)[0]

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

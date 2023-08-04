
import time
import pandas as pd
import numpy as np
import random
from common import Network as net
from common import  Seeker
from common import Vehicle
from common import KM_method
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
            vehicle = Vehicle.Vehicle(i, location,self.cfg)
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
        current_time_orders = self.order_list[self.order_list['beginTime_stamp'] > time_old]
        current_time_orders = current_time_orders[current_time_orders['beginTime_stamp'] <= self.time]
        self.current_seekers = []
        self.current_seekers_location = []
        for index, row in current_time_orders.iterrows():
            seeker = Seeker.Seeker( row)
            seeker.set_shortest_path(self.get_path(
                seeker.O_location, seeker.D_location))
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
            self.res['saved_ride_distance'] = np.sum(self.saved_travel_distance) - np.sum(self.total_travel_distance)

            self.res['platform_income'] = np.sum(self.platform_income)
            self.res['response_rate'] = len(list(set(self.his_order))) / len(self.order_list)
            self.res['carpool_rate'] = len(self.carpool_order) / len(self.his_order)

            # for system

            self.save_metric(path = 'output/system_metric.pkl')
            self.save_his_order(path = 'output/history_order.pkl')
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
            reward = self.batch_matching(takers, vehicles, seekers)
            end = time.time()
            # print('匹配用时{},time{},vehicles{},takers{},seekers{}'.format(end - start, self.time_slot, len(vehicles),len(takers) ,len(seekers)))
            # for vehicle in self.vehicle_list:
            #     print('vehicle.activate_time',vehicle.activate_time)
            #     print('vehicle.target',vehicle.target)
            self.total_reward += reward

            return reward,  False

    # 匹配算法
    def batch_matching(self, takers, vehicles, seekers):
        import time
        start = time.time()
        step_reward = 0
        # 构造权重矩阵
        demand = len(seekers)
        supply = len(takers) + len(vehicles)
        row_nums = demand + supply  # 加入乘客选择wait
        column_nums = demand + supply  # 加入司机选择wait
        # print('row_nums,column_nums ',row_nums,column_nums )
        dim = max(row_nums, column_nums)
        matrix = np.ones((dim, dim)) * -10000

        # # 从司机角度计算响应乘客的权重
        # for row in range(supply):
        #     for column in range(demand):

        #         if row < len(takers):
        #             start = time.time()
        #             matrix[row, column] = self.calTakersWeights(takers[row], seekers[column],
        #                                                         optimazition_target=self.optimazition_target,
        #                                                         matching_condition=self.matching_condition)
        #             end = time.time()
        #             # print('计算taker权重时间', end - start)

        #         else:
        #             start = time.time()
        #             matrix[row, column] = self.calVehiclesWeights(vehicles[row - len(takers)], seekers[column],
        #                                                           optimazition_target=self.optimazition_target,
        #                                                           matching_condition=self.matching_condition)
        #             end = time.time()
        #             # print('计算Vehicle权重时间', end - start)

        # 从乘客角度计算匹配权重
        for column in range(demand):
            # 当前seeker的zone
            location = seekers[column].O_location
            zone = self.network.Nodes[location].getZone()
            nodes = zone.nodes

            for row in range(supply):

                if row < len(takers):
                    if takers[row].location in nodes:
                        start = time.time()
                        matrix[row, column] = self.calTakersWeights(takers[row], seekers[column],
                                                                    optimazition_target=self.optimazition_target,
                                                                    matching_condition=self.matching_condition)
                        end = time.time()
                        # print('计算taker权重时间', end - start)
                    else:
                        continue
                    

                else:
                    if vehicles[row - len(takers)].location in nodes:
                        start = time.time()
                        matrix[row, column] = self.calVehiclesWeights(vehicles[row - len(takers)], seekers[column],
                                                                    optimazition_target=self.optimazition_target,
                                                                    matching_condition=self.matching_condition)
                        end = time.time()
                        # print('计算Vehicle权重时间', end - start)
                    else:
                        continue
                    



        # 计算司机选择调度的权重
        for row in range((row_nums - 1)):
            matrix[row, column_nums - 1] = 0
        
        # 计算乘客选择等待的权重
        for column in range(len(seekers)):
            for row in range(len(takers) + len(vehicles), row_nums):
                matrix[row, column] = - 3000

        end = time.time()
        # print('构造矩阵用时', end-start)
        # print(matrix)

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
        failed = 0
        successed = 0
        for i in range(len(takers)):
            #  第i个taker响应第res[1][i]个订单
            if res[i] >= len(seekers):
                # 接到了虚拟订单，taker应该进入下一个匹配池
                takers[i].reposition_target = 1
                # print('taker{}拼车失败，进入匹配池'.format(
                #     takers[i].id))
                failed += 1
            else:
                # 匹配到新乘客，需要先按原路线接到已有乘客
                if takers[i].location != takers[i].order_list[0].O_location:     
                    pickup_distance, takers[i].path, takers[i].path_length = self.network.get_path(
                        takers[i].location, takers[i].order_list[0].O_location)
                    takers[i].location = takers[i].order_list[0].O_location


                takers[i].order_list.append(seekers[res[i]])
                self.his_order.append(seekers[res[i]])
                self.carpool_order.append(seekers[res[i]])
                takers[i].reward += matrix[i, res[i]]
                step_reward += matrix[i, res[i]]
                # 记录目的地
                pickup_distance, takers[i].path, takers[i].path_length = self.network.get_path(
                    takers[i].location, takers[i].order_list[1].O_location)
                    
                # print('taker{}拼车成功，权重为{}'.format(
                #     takers[i].id, matrix[i, res[i]]))
                successed += 1
                # 记录seeker等待时间
                seekers[res[i]].set_waitingtime(
                    self.time - seekers[res[i]].begin_time_stamp)
                seekers[res[i]].response_target = 1

                # 做可视化记录，接单
                columns= ['vehicle_id','vehicle_type','depart_time','arrival_time','O_location','D_location','p1_id','p2_id'] 
                row = [takers[i].id, "pickup_p2", self.time - 0.1, self.time  + self.cfg.unit_driving_time * pickup_distance ,takers[i].location, seekers[res[i]].O_location,\
                    takers[i].order_list[0].id, takers[i].order_list[1].id ]
                self.visualization_df.loc[len(self.visualization_df.index)] = row
                # self.visualization_df.loc[len(self.visualization_df.index)] = row
                
                


        for i in range(len(vehicles)):
            #  第i个vehicle响应第res[1][i]个订单
            if res[i + len(takers)] >= len(seekers):
                # 接到了虚拟订单
                vehicles[i].reposition_target = 1
                # 原地等待时间超过上限
                if self.time - vehicles[i].activate_time > self.cfg.reposition_time_threshold:
                    # 调度
                    repostion_location = random.choice(self.locations)

                    distance, vehicles[i].path, vehicles[i].path_length = self.network.get_path(
                        vehicles[i].location, repostion_location )
                    # 做可视化记录，调度
                    columns= ['vehicle_id','vehicle_type','depart_time','arrival_time','O_location','D_location','p1_id','p2_id'] 
                    row = [vehicles[i].id, 'reposition', self.time-0.1 , self.time  + self.cfg.unit_driving_time * distance \
                            ,vehicles[i].location, repostion_location ,\
                            '', '' ]
                    self.visualization_df.loc[len(self.visualization_df.index)] = row
                else:
                    continue
                failed += 1

            else:
                # print('vehicle id{},order id{}'.format(vehicles[i].id, seekers[res[i + len(takers)]].id))
                vehicles[i].order_list.append(seekers[res[i + len(takers)]])
                self.his_order.append(seekers[res[i + len(takers)]])
                vehicles[i].reward += matrix[i +
                                             len(takers), res[i + len(takers)]]
                # 更新目的地
                pickup_distance, vehicles[i].path, vehicles[i].path_length \
                 = self.network.get_path(
                     vehicles[i].location, vehicles[i].order_list[0].O_location)
                # print('vehicles{}拼车成功，权重为{}'.
                #       format(vehicles[i].id, matrix[i + len(takers), res[i + len(takers)]]))
                successed += 1
                step_reward += matrix[i, res[i + len(takers)]]
                # 记录seeker等待时间
                seekers[res[i + len(takers)]].set_waitingtime(self.time -
                                                              seekers[res[i + len(takers)]].begin_time_stamp)
                seekers[res[i + len(takers)]].response_target = 1

                # 做可视化记录，接单
                columns= ['vehicle_id','vehicle_type','depart_time','arrival_time','O_location','D_location','p1_id','p2_id'] 
                row = [vehicles[i].id, 'pickup_p1', self.time - 0.1 , self.time  + self.cfg.unit_driving_time * pickup_distance,\
                     vehicles[i].location, seekers[res[i + len(takers)]].O_location,\
                    vehicles[i].order_list[0].id, '' ]
                self.visualization_df.loc[len(self.visualization_df.index)] = row


        # print('匹配时间{},匹配成功{},匹配失败{},takers{},vehicles{},demand{},time{}'.
        #       format(end-start, successed, failed, len(takers), len(vehicles), len(seekers), self.time_slot))
        start = time.time()



        # 更新位置
        for taker in takers:
            # 当前匹配没拼到新乘客
            if taker.reposition_target == 1:
                # 超出匹配时间
                if self.time - taker.order_list[0].begin_time_stamp - \
                        self.cfg.unit_driving_time * taker.p0_pickup_distance > 600:
                    # 已经10分钟还没拼到车了
                    # 派送时间
                    travel_distance = self.get_path(
                        taker.location, taker.order_list[0].D_location)

                    taker.drive_distance += travel_distance
                    taker.order_list[0].ride_distance = self.get_path(
                        taker.order_list[0].O_location, taker.order_list[0].D_location)
                    taker.order_list[0].shared_distance = 0
                    self.total_travel_distance += taker.order_list[0].ride_distance
                    self.saved_travel_distance += taker.order_list[0].ride_distance

                    self.platform_income.append(
                        taker.order_list[0].value -
                        self.cfg.unit_distance_cost/1000 * taker.drive_distance
                    )

                    # 更新智能体可以采取动作的时间
                    taker.activate_time += self.cfg.unit_driving_time * travel_distance

                    # 做可视化记录，派单
                    columns= ['vehicle_id','vehicle_type','depart_time','arrival_time','O_location','D_location','p1_id','p2_id'] 
                    row = [taker.id, 'failed_ridepooling,delivery_p1', self.time, taker.activate_time \
                        ,taker.order_list[0].O_location, taker.order_list[0].D_location,\
                        taker.order_list[0].id, '' ]
                    self.visualization_df.loc[len(self.visualization_df.index)] = row
                    
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


                else:
                    # 没超出匹配时间，根据当前目的地更新位置和时间
                    if taker.path:
                        # 没到目的地
                        taker.location = taker.path.pop()
                        taker.activate_time += self.cfg.unit_driving_time * taker.path_length.pop()
                    else:
                        # 到目的地了
                        taker.activate_time = self.time

            # 当前匹配有新乘客
            else:
                # 接新乘客
                pickup_distance, taker.path, taker.path_length \
                 = self.network.get_path(
                    taker.location, taker.order_list[1].O_location)
                
                pickup_time = self.cfg.unit_driving_time * pickup_distance
                taker.drive_distance += pickup_distance
                self.pickup_time.append(pickup_time)

                taker.order_list[0].ride_distance += pickup_distance
                    
                self.total_travel_distance += pickup_distance
                self.saved_travel_distance += taker.order_list[0].shortest_distance
                self.saved_travel_distance += taker.order_list[1].shortest_distance

                # 决定派送顺序，是否fifo
                fifo, distance = self.is_fifo(
                    taker.order_list[0], taker.order_list[1])
                if fifo:
                    # 先上先下
                    self.total_travel_distance += self.get_path(taker.order_list[1].O_location, taker.order_list[0].D_location)
                    self.total_travel_distance += self.get_path(taker.order_list[0].D_location, taker.order_list[1].D_location)

                    p0_invehicle = pickup_distance + distance[0]
                    p0_expected_distance = taker.order_list[0].shortest_distance
                    # 绕行
                    taker.order_list[0].set_detour(
                        p0_invehicle - p0_expected_distance)
                    p1_invehicle = sum(distance)
                    p1_expected_distance = taker.order_list[1].shortest_distance
                    taker.order_list[1].set_detour(
                        p1_invehicle - p1_expected_distance)
                    # travel time
                    taker.order_list[0].set_traveltime(
                        self.cfg.unit_driving_time * p0_invehicle)
                    taker.order_list[1].set_traveltime(
                        self.cfg.unit_driving_time * p1_invehicle)

                    # 做可视化记录，派单
                    columns= ['vehicle_id','vehicle_type','depart_time','arrival_time','O_location','D_location','p1_id','p2_id'] 
                    row = [taker.id, 'delivery_p1', self.time + self.cfg.unit_driving_time * pickup_distance, self.time + self.cfg.unit_driving_time *  p0_invehicle \
                            ,taker.order_list[1].O_location, taker.order_list[0].D_location,\
                            taker.order_list[0].id, taker.order_list[1].id ]
                    self.visualization_df.loc[len(self.visualization_df.index)] = row

                    row = [taker.id, 'delivery_p2', self.time + self.cfg.unit_driving_time *  p0_invehicle, self.time + self.cfg.unit_driving_time * (pickup_distance + sum(distance))\
                            ,taker.order_list[0].D_location, taker.order_list[1].D_location,\
                            taker.order_list[0].id, taker.order_list[1].id ]
                    self.visualization_df.loc[len(self.visualization_df.index)] = row


                                        

                else:
                    # 先上后下
                    self.total_travel_distance += self.get_path(taker.order_list[1].O_location, taker.order_list[1].D_location)
                    self.total_travel_distance += self.get_path(taker.order_list[1].D_location, taker.order_list[0].D_location)

                    p0_invehicle = pickup_distance + sum(distance)
                    p1_invehicle = distance[0]
                    p0_expected_distance = taker.order_list[0].shortest_distance
                    taker.order_list[0].set_detour(
                        p0_invehicle - p0_expected_distance)

                    taker.order_list[1].set_detour(
                        p1_invehicle - taker.order_list[1].shortest_distance)
                    # travel time
                    taker.order_list[0].set_traveltime(
                        self.cfg.unit_driving_time * p0_invehicle)
                    taker.order_list[1].set_traveltime(
                        self.cfg.unit_driving_time * p1_invehicle)

                    # 做可视化记录，派单
                    columns= ['vehicle_id','vehicle_type','depart_time','arrival_time','O_location','D_location','p1_id','p2_id'] 
                    row = [taker.id, 'delivery_p2', self.time + self.cfg.unit_driving_time * pickup_distance, self.time + self.cfg.unit_driving_time *  (pickup_distance + p1_invehicle) \
                            ,taker.order_list[1].O_location, taker.order_list[1].D_location,\
                            taker.order_list[0].id, taker.order_list[1].id ]
                    self.visualization_df.loc[len(self.visualization_df.index)] = row

                    row = [taker.id, 'delivery_p1', self.time + self.cfg.unit_driving_time * (pickup_distance + p1_invehicle),   self.time + self.cfg.unit_driving_time * p0_invehicle\
                            ,taker.order_list[1].D_location, taker.order_list[0].D_location,\
                            taker.order_list[0].id, taker.order_list[1].id ]
                    self.visualization_df.loc[len(self.visualization_df.index)] = row

                # 乘客的行驶距离
                taker.order_list[0].ride_distance = p0_invehicle
                taker.order_list[1].ride_distance = p1_invehicle
                taker.order_list[0].shared_distance = distance[0]
                taker.order_list[1].shared_distance = distance[0]



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
                dispatching_time = pickup_time +  self.cfg.unit_driving_time * sum(distance)
                taker.activate_time = self.time + dispatching_time
                # print('拼车完成，activate_time:{}'.format(taker.activate_time - self.time))
                # 更新智能体的位置
                taker.location = taker.order_list[1].D_location
                # 完成订单
                taker.order_list = []
                taker.target = 0  # 变成vehicle
                taker.drive_distance = 0
                taker.reward = 0
        
                

        for vehicle in vehicles:
            if vehicle.reposition_target == 1:
                if vehicle.path:
                    # 没到目的地
                    if self.time - vehicle.activate_time > self.cfg.unit_driving_time *vehicle.path_length[0] :
                        vehicle.location =  vehicle.path.pop()
                        vehicle.path_length.pop()
                        vehicle.activate_time = self.time 

                else:
                    # 到目的地了
                    vehicle.activate_time = self.time 

            else:
                vehicle.target = 1  # 变成taker
                vehicle.origin_location = vehicle.location
                # 接新乘客
                pickup_distance, vehicle.path, vehicle.path_length = self.network.get_path(
                    vehicle.location, vehicle.order_list[0].O_location)

                if vehicle.path:
                    if self.time - vehicle.activate_time > self.cfg.unit_driving_time * vehicle.path_length[0] :
                        vehicle.location =  vehicle.path.pop()
                        vehicle.path_length.pop()
                        vehicle.activate_time = self.time 
                        vehicle.p0_pickup_distance += pickup_distance
                        vehicle.drive_distance += pickup_distance
                else:
                    # 接到乘客了
                    pickup_time = self.cfg.unit_driving_time * vehicle.p0_pickup_distance
                    self.pickup_time.append(pickup_time)    
                    vehicle.location = vehicle.order_list[0].O_location

        end = time.time()
        # print('派送用时{},takers{},vehicles{}'.format(end-start, len(takers), len(vehicles)))

        self.remain_seekers = []
        for seeker in seekers:
            if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < self.cfg.delay_time_threshold:
                seeker.set_delay(self.time)
                self.remain_seekers.append(seeker)

        return step_reward


    def calTakersWeights(self, taker, seeker,  optimazition_target, matching_condition):
        # expected shared distance
        pick_up_distance = self.get_path(
            seeker.O_location, taker.location)
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
            p1_invehicle = seeker.shortest_distance
            p0_detour = p0_invehicle - taker.order_list[0].shortest_distance
            p1_detour = 0

        if matching_condition and (pick_up_distance > self.pickup_distance_threshold or
                                   p0_detour > self.detour_distance_threshold or
                                   p1_detour > self.detour_distance_threshold):
            # print('detour_distance not pass', detour_distance)
            return self.cfg.dead_value
        else:
            reward = (seeker.shortest_distance + taker.order_list[0].shortest_distance
                      - (p0_invehicle + p1_invehicle - shared_distance) - pick_up_distance) 
            # print('taker reward',reward)
            return reward


    def calVehiclesWeights(self, vehicle, seeker,  optimazition_target, matching_condition):

        pick_up_distance = self.get_path(
            vehicle.location, seeker.O_location)
        if matching_condition and (pick_up_distance > self.cfg.pickup_distance_threshold):
            return self.cfg.dead_value
        else:

            reward = (0- pick_up_distance) 
            # print('vacant vehicle reward',reward)
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

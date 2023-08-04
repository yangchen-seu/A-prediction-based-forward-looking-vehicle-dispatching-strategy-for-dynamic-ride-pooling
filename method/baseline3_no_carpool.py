

import time
import pandas as pd
import numpy as np
import random
from ..common import Network as net
from ..common import  Seeker
from ..common import Vehicle
from ..common import KM_method
import os

 
class Simulation():
 
   
    def __init__(self, cfg) -> None:
        self.date = cfg.date
        self.cfg = cfg
        self.order_list = pd.read_csv(
            self.cfg.order_file).sample(frac=cfg.demand_ratio, random_state=1)
        self.vehicle_num = int(len(self.order_list) / cfg.order_driver_ratio)
        print('cfg.order_driver_ratio',cfg.order_driver_ratio,'len(self.order_list) ',len(self.order_list) )
        print('self.vehicle_num',self.vehicle_num)
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

        self.takers = []
        self.current_seekers = []  # 存储需要匹配的乘客
        self.remain_seekers = []
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
        self.dispatch_time = []
        self.platform_income = []
        self.shared_distance = []
        self.reposition_time = []
        self.total_travel_distance = 0
        self.saved_travel_distance = 0

        self.carpool_order = []
        self.ride_distance_error = []
        self.shared_distance_error = []

 

    def time_reset(self):
        #转换成时间数组
        self.time  = time.strptime(self.cfg.date +  self.cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S")
        #转换成时间戳
        self.time  = time.mktime(self.time )
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
            self.current_seekers_location.append(seeker.O_location)
        for seeker in self.remain_seekers:
            self.current_seekers.append(seeker)
            self.current_seekers_location .append(seeker.O_location)

        start = time.time()
        reward, done = self.process( self.time)
        end = time.time()
        print('匹配用时{},time{}'.format(end - start, self.time_slot))

        return reward,  done
       
 
    #
    def process(self, time_, ):
        reward = 0
        vehicles = []
        seekers = self.current_seekers
 
        if self.time >= time.mktime(time.strptime(self.cfg.date + self.cfg.simulation_end_time, "%Y-%m-%d %H:%M:%S") ) :
            print('当前episode仿真时间结束,奖励为:', self.total_reward)

            # 计算系统指标
            self.res = {}
            # for seekers
            for order in self.his_order:
                self.waitingtime.append(order.waitingtime)
                self.traveltime.append(order.traveltime)

            self.res['waitingTime'] = np.mean(self.waitingtime)
            self.res['traveltime'] = np.mean(self.traveltime)

            # for vehicle
            self.res['pickup_time'] = np.mean(self.pickup_time)
            self.res['dispatch_time'] = np.mean(self.dispatch_time)
            self.res['total_ride_distance'] = np.sum(self.total_travel_distance)

            self.res['platform_income'] = np.sum(self.platform_income)
            self.res['response_rate'] = len(list(set(self.his_order))) / len(self.order_list)
            print('his_order{},total_order{}'.format(len(list(set(self.his_order))) , len(self.order_list)))


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
                if vehicle.state == 1 : # 能执行动作
                    vehicles.append(vehicle)
            start = time.time()
            reward = self.batch_matching(vehicles, seekers)
            end = time.time()           
            print('匹配用时{},time{},vehicles{},seekers{}'.format(end - start, self.time_slot, len(vehicles) ,len(seekers))) 
            # for vehicle in self.vehicle_list:
            #     print('vehicle.activate_time',vehicle.activate_time)
            #     print('vehicle.target',vehicle.target)
            self.total_reward += reward

            return reward,  False
 
    # 匹配算法
    def batch_matching(self, vehicles, seekers):
        
        # 构造权重矩阵
        supply = len(vehicles)
        demand = len(seekers)
        row_nums = demand + supply  # 加入乘客选择wait
        column_nums = demand + supply  # 加入司机选择wait
        # print('row_nums,column_nums ',row_nums,column_nums )
        dim = max(row_nums, column_nums)
        matrix = np.ones((dim, dim)) * self.cfg.dead_value
 
        # for row in range(row_nums):
        #     for column in range(column_nums):
        #         matrix[row,column] = self.calVehiclesWeights(vehicles[row], seekers[column],\
        #                 optimazition_target = self.optimazition_target, \
        #         matching_condition = self.matching_condition )

        # 从乘客角度计算匹配权重
        for column in range(len(seekers)):
            # 当前seeker的zone
            location = seekers[column].O_location
            zone = self.network.Nodes[location].getZone()
            nodes = zone.nodes

            for row in range(len(vehicles)):
                if vehicles[row].location in nodes:
                    start = time.time()
                    matrix[row, column] = self.calVehiclesWeights(vehicles[row], seekers[column],
                                                                    optimazition_target=self.optimazition_target,
                                                                    matching_condition=self.matching_condition)
                    end = time.time()
                    # print('计算taker权重时间', end - start)

        
        # 计算乘客不被响应的权重
        for column in range(len(seekers), dim):
            for row in range( len(vehicles), dim):
                matrix[row, column] = - 3000

        # 匹配
        if row_nums == 0  or  column_nums == 0:
            return 0
        matcher = KM_method.KM_method(matrix)
        res, weights = matcher.run()
        # print(res)
        for i in range(len(vehicles)):
            #  第i个taker响应第res[1][i]个订单
            if res[i] >= len(seekers):
                # 接到了虚拟订单，taker应该进入下一个匹配池
                vehicles[i].reposition_target = 1
            else:
                vehicles[i].order_list.append(seekers[res[i]])
                self.his_order.append(seekers[res[i]])
                vehicles[i].reward += matrix[i,res[i]]
                # 记录seeker等待时间
                seekers[res[i]].set_waitingtime(self.time - seekers[res[i]].begin_time_stamp)
                seekers[res[i]].response_target = 1

 
        # 更新位置
        for vehicle in vehicles:
            if vehicle.reposition_target == 1:
                if vehicle.path:
                    # 没到目的地
                    if self.time - vehicle.activate_time > self.cfg.unit_driving_time * vehicle.path_length[0]:
                        vehicle.location = vehicle.path.pop()
                        vehicle.path_length.pop()
                        vehicle.activate_time = self.time

                else:
                    # 到目的地了
                    vehicle.activate_time = self.time
            else:

                # 接新乘客
                pickup_distance, vehicle.path, vehicle.path_length = self.network.get_path(
                    vehicle.location, vehicle.order_list[0].O_location)

                if vehicle.path:
                    if self.time - vehicle.activate_time > self.cfg.unit_driving_time * vehicle.path_length[0]:
                        vehicle.location = vehicle.path.pop()
                        vehicle.path_length.pop()
                        vehicle.activate_time = self.time
                        vehicle.p0_pickup_distance += pickup_distance
                        vehicle.drive_distance += pickup_distance

                else:
                    # 接到乘客了
                    pickup_time = self.cfg.unit_driving_time * vehicle.p0_pickup_distance
                    self.pickup_time.append(pickup_time)
                    vehicle.location = vehicle.order_list[0].O_location
                    self.platform_income.append(vehicle.order_list[0].value - self.cfg.unit_distance_cost/1000 * (vehicle.p0_pickup_distance + vehicle.order_list[0].shortest_distance))
                    self.total_travel_distance += vehicle.order_list[0].shortest_distance

                    vehicle.activate_time += (pickup_time + self.cfg.unit_driving_time *vehicle.order_list[0].shortest_distance)
                    # 完成订单
                    vehicle.order_list = []
                    vehicle.drive_distance = 0
                    vehicle.reward = 0
                    vehicle.path = []
                    vehicle.path_length = []

        self.remain_seekers = []
        for seeker in seekers:
            if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < 600 :
                seeker.set_delay(self.time)
                self.remain_seekers.append(seeker)

        return 0
 


    def calVehiclesWeights(self, vehicle, seeker,  optimazition_target, matching_condition):
        if optimazition_target == 'platform_income':
            dispatch_distance = self.get_path(seeker.O_location, seeker.D_location)
            pick_up_distance = self.get_path(seeker.O_location, vehicle.location)
            if matching_condition and (pick_up_distance > self.cfg.pickup_distance_threshold \
                or dispatch_distance - pick_up_distance < 0):
                # print('vehicle pick_up_distance not pass', pick_up_distance)
                return self.cfg.dead_value
            else:
                reward = self.cfg.unit_distance_value/1000 * (dispatch_distance - pick_up_distance)
                return reward            


    def is_fifo(self,p0,p1):
        fifo = [self.get_path(p1.O_location, p0.D_location) ,
                self.get_path(p0.D_location,p1.D_location)]
        lifo = [self.get_path(p1.O_location, p1.D_location) ,
                self.get_path(p1.D_location,p0.D_location)]
        if fifo < lifo:
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

import time
import pandas as pd
import numpy as np
from ..common import Network as net
from ..common  import Config
from ..common import  Seeker
from ..common import Vehicle
import random
 
Config = Config.Config()
 
class Simulation():
 
    def __init__(self, cfg) -> None:
        self.date = cfg.date
        self.order_list = pd.read_csv(
            './input/order.csv').sample(frac=cfg.demand_ratio, random_state=1)
        self.vehicle_num = int(len(self.order_list) / 4)
        self.order_list['beginTime_stamp'] = self.order_list['dwv_order_make_haikou_1.departure_time'].apply(
            lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        self.begin_time = time.mktime(time.strptime(
            cfg.date + cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))
        self.end_time = time.mktime(time.strptime(
            cfg.date + cfg.simulation_end_time, "%Y-%m-%d %H:%M:%S"))
        self.locations = self.order_list['O_location'].unique()
        self.network = net.Network()
        self.shortest_path = pd.read_csv('./input/shortest_path.csv')

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
            vehicle = Vehicle.Vehicle(i, location)
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

    def reset(self):
        self.takers = []
        self.current_seekers = []  # 存储需要匹配的乘客
        self.remain_seekers = []
        self.vehicle_list = []
        self.total_reward = 0
        self.order_list = pd.read_csv(
            './input/order.csv').sample(frac=Config.demand_ratio, random_state=1)
        self.order_list['beginTime_stamp'] = self.order_list['dwv_order_make_haikou_1.arrive_time'].apply(
            lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        self.begin_time = time.mktime(time.strptime(
            Config.date + Config.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))
        self.end_time = time.mktime(time.strptime(
            Config.date + Config.simulation_end_time, "%Y-%m-%d %H:%M:%S"))
        self.order_list = self.order_list[self.order_list['beginTime_stamp']
                                          >= self.begin_time]
        self.order_list = self.order_list[self.order_list['beginTime_stamp']
                                          <= self.end_time]

        self.time_reset()

        for i in range(self.vehicle_num):
            random.seed(i)
            location = random.choice(self.locations)
            vehicle = Vehicle.Vehicle(i, location)
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
        self.time  = time.strptime(Config.date +  Config.simulation_begin_time, "%Y-%m-%d %H:%M:%S")
        #转换成时间戳
        self.time  = time.mktime(self.time )
        self.time_slot = 0

 
    def step(self,):
        time_old = self.time
        self.time += self.time_unit
        self.time_slot += 1

        # 筛选时间窗内的订单
        current_time_orders = self.order_list[self.order_list['beginTime_stamp'] >time_old]
        current_time_orders = current_time_orders[current_time_orders['beginTime_stamp'] <= self.time]
        self.current_seekers = [] # 暂时不考虑等待的订单
        self.current_seekers_location = []
        for index, row in current_time_orders.iterrows():
            seeker = Seeker.Seeker(index, row)
            seeker.set_shortest_path(self.get_path(seeker.O_location, seeker.D_location))
            value = Config.unit_distance_value /1000 * seeker.shortest_distance
            seeker.set_value(value)
            self.current_seekers.append(seeker)
            self.current_seekers_location .append(seeker.O_location)
        for seeker in self.remain_seekers:
            self.current_seekers.append(seeker)
            self.current_seekers_location .append(seeker.O_location)

        reward, done = self.process( self.time)
        self.total_reward += reward
        return reward, done
       
 
    #
    def process(self, time_, ):
        reward = 0
        takers = []
        vehicles = []
        seekers = self.current_seekers

        if self.time >= time.mktime(time.strptime(Config.date + Config.simulation_end_time, "%Y-%m-%d %H:%M:%S")):
            print('当前episode仿真时间结束,奖励为:', self.total_reward)

            # 计算系统指标
            self.res = {}
            # for seekers
            for order in self.his_order:
                self.waitingtime.append(order.waitingtime)
                self.traveltime.append(order.traveltime)
                self.ride_distance_error.append(abs(order.rs- order.ride_distance) )
                self.shared_distance_error.append(abs(order.es- order.shared_distance) )
            for order in self.carpool_order:
                self.detour_distance.append(order.detour)
            self.res['waitingTime'] = np.mean(self.waitingtime)
            self.res['traveltime'] = np.mean(self.traveltime)
            self.res['detour_distance'] = np.mean(self.detour_distance)

            # for vehicle
            self.res['pickup_time'] = np.mean(self.pickup_time)
            self.res['dispatch_time'] = np.mean(self.dispatch_time)
            self.res['shared_distance'] = np.mean(self.shared_distance)
            self.res['total_ride_distance'] = np.sum(self.total_travel_distance)
            self.res['saved_ride_distance'] = np.sum(self.saved_travel_distance) - np.sum(self.total_travel_distance)

            self.res['platform_income'] = np.sum(self.platform_income)
            self.res['response_rate'] = len(list(set(self.his_order))) / len(self.order_list)
            self.res['carpool_rate'] = len(
                self.carpool_order) / len(self.his_order)
            self.res['ride_distance_error'] = self.ride_distance_error
            self.res['shared_distance_error'] = self.shared_distance_error

            return reward, True
        else:
            # print('当前episode仿真时间:',time_)
            # 判断智能体是否能执行动作
            for vehicle in self.vehicle_list:
                # print('vehicle.activate_time',vehicle.activate_time)
                # print('vehicle.state',vehicle.state)
                vehicle.is_activate(time_)
                vehicle.reset_reposition()
                if vehicle.state == 1 : # 能执行动作
                    if vehicle.target == 0:
                        vehicles.append(vehicle)
                    else:
                        takers.append(vehicle)
            # print('len(vehicles)',len(vehicles),'len(takers)', len(takers))
            start = time.time()
            reward = self.first_protocol_matching(takers, vehicles, seekers)
            end = time.time()
            print('匹配用时{},time{}'.format(end - start, self.time_slot))
            
            # for vehicle in self.vehicle_list:
            #     print('vehicle.activate_time',vehicle.activate_time)
           
            return reward, False
 
 
 
    # 匹配算法
    def first_protocol_matching(self, takers, vehicles, seekers):
        step_rewrad = 0
        seekers = seekers
        if len(seekers) == 0:
            return 0

        for seeker in seekers:
            seeker.wait_utility = self.calSeekerWaitingWeights(seeker, optimazition_target = 'expected_shared_distance')

        for taker in takers:
            reward, seekers = self.assign_taker(taker, seekers)

            step_rewrad += reward
            self.total_reward += reward
        
        for vehicle in vehicles:
            reward, seekers = self.assign_vehicle(vehicle, seekers)

            
            step_rewrad += reward
            self.total_reward += reward
 
        # 更新位置
        for taker in takers:
            if taker.reposition_target == 1:
                if self.time - taker.order_list[0].begin_time_stamp - \
                        Config.unit_driving_time * taker.p0_pickup_distance > 600:
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
                        taker.order_list[0].value - Config.unit_distance_cost/1000 * taker.drive_distance
                                                )

                    # 更新智能体可以采取动作的时间
                    taker.activate_time += Config.unit_driving_time * travel_distance
                    # print('没有拼到车，activate_time:{}'.format(taker.activate_time - self.time))
                    # 更新智能体的位置
                    taker.location = taker.order_list[0].D_location

                    # 完成订单
                    taker.order_list = []
                    taker.target = 0  # 变成vehicle
                    taker.drive_distance = 0
                    taker.reward = 0

                else:
                    # 没超出匹配时间，根据当前目的地更新位置和时间
                    if taker.path:
                        # 没到目的地
                        taker.location = taker.path.pop()
                        taker.activate_time += self.cfg.unit_driving_time * taker.path_length.pop()
                    else:
                        # 到目的地了
                        taker.activate_time = self.time
            else:
                # 接驾时间
                pickup_distance = self.get_path(
                    taker.order_list[1].O_location, taker.location)

                self.total_travel_distance += pickup_distance
                self.saved_travel_distance += taker.order_list[0].shortest_distance
                self.saved_travel_distance += taker.order_list[1].shortest_distance

                pickup_time = Config.unit_driving_time * pickup_distance
                taker.drive_distance += pickup_distance
                self.pickup_time.append(pickup_time)

                # 决定派送顺序，是否fofo
                fofo, distance = self.is_fofo(
                    taker.order_list[0], taker.order_list[1])
                if fofo:
                    # 先上先下
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
                        Config.unit_driving_time * p0_invehicle)
                    taker.order_list[0].ride_distance = p0_invehicle

                    taker.order_list[1].set_traveltime(
                        Config.unit_driving_time * p1_invehicle)
                    taker.order_list[1].ride_distance = p1_invehicle
                    
                    # shared distance
                    taker.order_list[0].shared_distance = distance[0]
                    taker.order_list[1].shared_distance = distance[0]               
                    # print('先上先下,p0_invehicle{},p0detour{}, p1_invehicle{},p1detour{}'\
                    #     .format(p0_invehicle, taker.order_list[0].detour, p1_invehicle, taker.order_list[1].detour))
                else:
                    # 先上后下
                    p0_invehicle = pickup_distance + sum(distance)
                    p1_invehicle = distance[0]
                    p0_expected_distance = taker.order_list[0].shortest_distance
                    taker.order_list[0].set_detour(
                        p0_invehicle - p0_expected_distance)
                    
                    taker.order_list[1].set_detour(p1_invehicle - taker.order_list[1].shortest_distance)
                    # travel time
                    taker.order_list[0].set_traveltime(
                        Config.unit_driving_time * p0_invehicle)
                    taker.order_list[1].set_traveltime(
                        Config.unit_driving_time * p1_invehicle)
                    # print('先上后下,p0_invehicle{},p0detour{}, p1_invehicle{},p1detour{}'\
                    #     .format(p0_invehicle, taker.order_list[0].detour, p1_invehicle, taker.order_list[1].detour))

                # 计算司机完成两个订单需要的时间
                dispatching_time = pickup_time + \
                    Config.unit_driving_time * sum(distance)
                self.dispatch_time.append(dispatching_time)

                self.total_travel_distance += sum(distance)

                taker.drive_distance += sum(distance)
                # 计算平台收益
                self.platform_income.append(
                        Config.discount_factor * (taker.order_list[0].value + taker.order_list[1].value ) - \
                            Config.unit_distance_cost/1000 * taker.drive_distance
                                                )

                # 计算拼车距离
                self.shared_distance.append(distance[0])

                # 完成该拼车过程所花费的时间
                time_consume = dispatching_time
                # 更新智能体可以采取动作的时间
                taker.activate_time += time_consume
                # print('拼车完成，activate_time:{}'.format(taker.activate_time - self.time))
                # 更新智能体的位置
                taker.location = taker.order_list[1].D_location
                # 完成订单
                taker.order_list = []
                taker.target = 0  # 变成vehicle
                taker.reward = 0
 
 
        for vehicle in vehicles:
            if vehicle.reposition_target == 1 :
                if  self.time - vehicle.activate_time > Config.reposition_time_threshold:
                    # 调度
                    repostion_location = random.choice(
                        self.current_seekers_location)
                    reposition_time = Config.unit_driving_time * \
                        self.get_path(
                            vehicle.location, repostion_location)
                    vehicle.location = repostion_location
                    vehicle.activate_time += reposition_time
                    self.reposition_time.append(reposition_time)
                else:
                    continue
            else:
                vehicle.target = 1  # 变成taker
                distance = self.network.get_path(
                    vehicle.location, vehicle.order_list[0].O_location)
                vehicle.p0_pickup_distance = distance
                vehicle.drive_distance += distance
                pickup_time = Config.unit_driving_time * distance

                vehicle.origin_location = vehicle.location

                vehicle.location = vehicle.order_list[0].O_location
                self.pickup_time.append(pickup_time)
                vehicle.activate_time += pickup_time


        self.remain_seekers = []
        for seeker in seekers:
            if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < Config.delay_time_threshold:
                seeker.set_delay(self.time)
                self.remain_seekers.append(seeker)

        return step_rewrad
 
 
 
 

    def assign_taker(self, taker, seekers):
        if not seekers:
            taker.reposition_target = 1
            return 0 , []
        match = {}
        for seeker in seekers:
            match[seeker] = self.calTakersWeights(seeker, taker, optimazition_target = self.optimazition_target, \
                matching_condition = self.matching_condition)
        reward = max(match.values())
        for key,value in match.items():
            if(value == max(match.values())):
                # print('value',value,key.wait_utility,'key.wait_utility')
                if value ==  Config.dead_value or value < key.wait_utility:
                    taker.reposition_target = 1
                    break
                taker.order_list.append(key)
                self.his_order.append(key)
                self.carpool_order.append(key)
                # print('taker:{},order:{}'.format(taker.id,key.id))
                key.set_waitingtime(
                    self.time - key.begin_time_stamp)
                seekers.remove(key)
                break
        return reward, seekers

    def calTakersWeights(self, seeker, taker, optimazition_target, matching_condition):
        if optimazition_target == 'platform_income':
            dispatch_distance = self.get_path(
                seeker.O_location, seeker.D_location)
            pick_up_distance = self.get_path(
                seeker.O_location, taker.location)
            if matching_condition and (pick_up_distance > self.pickup_distance_threshold):
                # print('taker pick_up_distance not pass', pick_up_distance)
                return Config.dead_value
            else:
                # print('taker pick_up_distance', pick_up_distance,'dispatch_distance',dispatch_distance)
                reward = Config.unit_distance_value/1000 * dispatch_distance * seeker.delay
                return reward

        else:  # expected shared distance
            pick_up_distance = self.get_path(
                seeker.O_location, taker.order_list[0].O_location)
            fofo, distance = self.is_fofo(taker.order_list[0], seeker)

            if fofo:
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
                                       p1_detour> self.detour_distance_threshold):
                # print('detour_distance not pass', detour_distance)
                return Config.dead_value
            else:
                reward = (shared_distance - pick_up_distance) * seeker.delay

                return reward


    def assign_vehicle(self, vehicle, seekers):
        if not seekers:
            vehicle.reposition_target = 1
            return 0 , []
        match = {}
        for seeker in seekers:
            match[seeker] = self.calVehiclesWeights(vehicle,  seeker, optimazition_target = self.optimazition_target, \
                matching_condition = self.matching_condition)
        reward = max(match.values())
        for key,value in match.items():
            if(value == max(match.values())):
                if value ==  Config.dead_value or value < key.wait_utility:
                    vehicle.reposition_target = 1
                    break
                vehicle.order_list.append(key)
                self.his_order.append(key)
                # print('vehicle:{},order:{}'.format(vehicle.id,key.id))
                key.set_waitingtime(
                    self.time - key.begin_time_stamp)
                seekers.remove(key)
                break
        return reward, seekers


    def calVehiclesWeights(self, vehicle, seeker,  optimazition_target, matching_condition):
        if optimazition_target == 'platform_income':
            dispatch_distance = self.get_path(
                seeker.O_location, seeker.D_location)
            pick_up_distance = self.get_path(
                seeker.O_location, vehicle.location)
            if matching_condition and (pick_up_distance > Config.pickup_distance_threshold
                                       or dispatch_distance - pick_up_distance < 0):
                # print('vehicle pick_up_distance not pass', pick_up_distance)
                return Config.dead_value
            else:
                reward = Config.unit_distance_value/1000 * \
                    (dispatch_distance - pick_up_distance) * seeker.delay
                return reward

        else:  # expected shared distance
            pick_up_distance = self.get_path(
                seeker.O_location, vehicle.location)
            if matching_condition and (pick_up_distance > Config.pickup_distance_threshold):
                return Config.dead_value
            else:
                reward = 0 - pick_up_distance
                return reward

    # 计算乘客选择等待的权重
    def calSeekerWaitingWeights(self, seeker,  optimazition_target):
        if optimazition_target == 'platform_income':
            # 不可行
            return seeker.delay

        else:  # expected shared distance
            gamma = 10
            reward = 0 - gamma * seeker.k * 60
            
            return reward

    def get_path(self, O, D):
        tmp = self.shortest_path[(self.shortest_path['O'] == O) & (
            self.shortest_path['D'] == D)]
        return tmp['distance'].values[0]

    def is_fofo(self, p0, p1):
        fofo = [self.get_path(p1.O_location, p0.D_location),
                self.get_path(p0.D_location, p1.D_location)]
        lofo = [self.get_path(p1.O_location, p1.D_location),
                self.get_path(p1.D_location, p0.D_location)]
        if sum(fofo) < sum(lofo):
            return True, fofo
        else:
            return False, lofo

    def save_metric(self, path="output/system_metric.pkl"):
        dic = {}
        dic['pickup_time'] = self.pickup_time
        dic['detour_distance'] = self.detour_distance
        dic['traveltime'] = self.traveltime
        dic['waiting_time'] = self.waitingtime

        dic['dispatch_time'] = self.dispatch_time
        dic['platform_income'] = self.platform_income
        dic['shared_distance'] = self.shared_distance
        dic['response_rate'] = self.res['response_rate']
        dic['carpool_rate'] = self.res['carpool_rate']
        dic['total_ride_distance'] = self.total_travel_distance
        dic['saved_distance'] = self.saved_travel_distance - self.total_travel_distance
        dic['ride_distance_error'] = self.res['ride_distance_error'] 
        dic['shared_distance_error'] = self.res['shared_distance_error']   



        import pickle

        with open(path, "wb") as tf:
            pickle.dump(dic,tf)
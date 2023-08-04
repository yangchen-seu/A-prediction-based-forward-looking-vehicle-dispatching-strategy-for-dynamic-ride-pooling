
from multiprocessing.connection import wait
import time
import pandas as pd
import numpy as np
import Network as net
import Config
import  Seeker
import Vehicle
import random
from common import Hungarian, KM_method

 
Config = Config.Config()
 
class Simulation():
 
   
    def __init__(self, agent_lis, critic, cfg ) -> None:
        self.date = cfg.date
        self.order_list = pd.read_csv('input\\order.csv').sample(frac= cfg.demand_ratio, random_state=1)
        self.order_list['beginTime_stamp'] = self.order_list['dwv_order_make_haikou_1.arrive_time'].apply(lambda x:time.mktime(time.strptime(x,'%Y-%m-%d %H:%M:%S')))
        self.critic = critic 
        self.begin_time  = time.mktime(time.strptime(cfg.date +  cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))
        self.end_time  = time.mktime(time.strptime(cfg.date +  cfg.simulation_end_time, "%Y-%m-%d %H:%M:%S"))
        self.locations = self.order_list['O_location'].unique()
        self.network = net.Network()
        self.time_unit = 10 # 控制的时间窗,每10s匹配一次
        self.index = 0 # 计数器
        self.total_reward = 0
        self.optimazition_target = cfg.optimazition_target # 仿真的优化目标
        self.matching_condition = cfg.matching_condition # 匹配时是否有条件限制
 
        self.vehicle_list = agent_lis # 存储所有的vehicle
        self.takers = []
        self.current_seekers = [] # 存储需要匹配的乘客
        self.time_reset()


        # system metric
        self.his_order = [] # all orders responsed
        self.taker_pickup_time = [] 
        self.extra_distance = []
        self.saved_distance = [] 
        self.waiting_time = []
        self.dispatch_time = []
        self.success_rate = 0
        self.platflorm_income = []
        self.shared_distance = []


            
       
    def reset(self):
        self.takers = []
        self.current_seekers = [] # 存储需要匹配的乘客
        self.total_reward = 0
        self.order_list = pd.read_csv('input\\order.csv').sample(frac= Config.demand_ratio, random_state=1)
        self.order_list['beginTime_stamp'] = self.order_list['dwv_order_make_haikou_1.arrive_time'].apply(lambda x:time.mktime(time.strptime(x,'%Y-%m-%d %H:%M:%S')))
        self.begin_time  = time.mktime(time.strptime(Config.date +  Config.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))
        self.end_time  = time.mktime(time.strptime(Config.date +  Config.simulation_end_time, "%Y-%m-%d %H:%M:%S"))
        self.order_list = self.order_list[self.order_list['beginTime_stamp'] >= self.begin_time ]
        self.order_list = self.order_list[self.order_list['beginTime_stamp'] <= self.end_time]
        self.time_reset()
        
        # system metric
        self.his_order = [] # all orders responsed
        self.taker_pickup_time = [] 
        self.extra_distance = []
        self.saved_distance = [] 
        self.waiting_time = []
        self.dispatch_time = []
        self.success_rate = 0
        self.platflorm_income = []
        self.shared_distance = []
 

    def time_reset(self):
        #转换成时间数组
        self.time  = time.strptime(Config.date +  Config.simulation_begin_time, "%Y-%m-%d %H:%M:%S")
        #转换成时间戳
        self.time  = time.mktime(self.time )
        self.time_slot = 0 
        print('time reset:', self.time)
 
    def step(self,):
        time_old = self.time
        self.time += self.time_unit
        self.time_slot += 1

        # 筛选时间窗内的订单
        current_time_orders = self.order_list[self.order_list['beginTime_stamp'] >= time_old]
        current_time_orders = current_time_orders[current_time_orders['beginTime_stamp'] <= self.time]
        self.current_seekers = [] # 暂时不考虑等待的订单
        for index, row in current_time_orders.iterrows():
            seeker = Seeker.Seeker(index, row)
            self.current_seekers.append(seeker)
   
        reward, done = self.process( self.time)

        return reward, self.time_slot, done
       
 
    #
    def process(self, time_, ):
        reward = 0
        takers = []
        vehicles = []
        seekers = self.current_seekers
 
        if self.time >= time.mktime(time.strptime(Config.date + Config.simulation_end_time, "%Y-%m-%d %H:%M:%S") ) :
            print('当前episode仿真时间结束,奖励为:', self.total_reward)
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
                    # print('id{},vehicle.target{}'.format(vehicle.id, vehicle.target))
                    if vehicle.target == 0:
                        vehicles.append(vehicle)
                    else:
                        # print('id{},vehicle.target{}'.format(vehicle.id, vehicle.target))
                        takers.append(vehicle)
            # print('len(vehicles)',len(vehicles),'len(takers)', len(takers))
            reward = self.batch_matching(takers, vehicles, seekers)
            # for vehicle in self.vehicle_list:
            #     print('vehicle.activate_time',vehicle.activate_time)
            #     print('vehicle.target',vehicle.target)

            return reward, False
 
    # 匹配算法
    def batch_matching(self, takers, vehicles, seekers):
 
        # 构造权重矩阵
        row_nums = len(takers) + len(vehicles)
        column_nums = len(seekers)
        # print('row_nums,column_nums ',row_nums,column_nums )s
        dim = max(row_nums, column_nums)
        matrix = np.zeros((dim, dim))
 
        for row in range(row_nums):
            for column in range(column_nums):
                if row < len(takers):
                    matrix[row,column] = self.calTakersWeights(takers[row], seekers[column],\
                        optimazition_target = self.optimazition_target, \
                matching_condition = self.matching_condition )
 
                else:
                    matrix[row,column] = self.calVehiclesWeights(vehicles[row - len(takers)] , seekers[column], \
                        optimazition_target = self.optimazition_target, \
                matching_condition = self.matching_condition)
        # print(matrix)
 
        # 匹配
        if row_nums == 0 or  column_nums == 0:
            return 0
        matcher = KM_method.KM_method(matrix)
        res, weights = matcher.run()
        print('res',res)
        print('weights',weights)
        reward = 0
        for i in range(len(takers)):
            #  第i个taker响应第res[1][i]个订单
            if res[i] >= len(seekers):
                # 接到了虚拟订单，taker应该进入下一个匹配池
                takers[i].reposition_target = 1
            else:
                takers[i].order_list.append(seekers[res[i]])
                self.his_order.append(seekers[res[i]])
                takers[i].reward += matrix[i,res[i]]
                reward += matrix[i,res[i]]
                self.total_reward += reward

           
           
        for i in range(len(vehicles)):
            #  第i个vehicle响应第res[1][i]个订单
            if res[i + len(takers)] >= len(seekers):
                # 接到了虚拟订单，taker应该进入下一个匹配池
                vehicles[i].reposition_target = 1
                # print('vehicle id{}'.format(vehicles[i].id))

            else:
                # print('vehicle id{},order id{}'.format(vehicles[i].id, seekers[res[i + len(takers)]].id))
                vehicles[i].order_list.append(seekers[res[i + len(takers)]])
                self.his_order.append(seekers[res[i + len(takers)]])
                vehicles[i].reward += matrix[i + len(takers),res[i + len(takers)]]
                reward += matrix[i,res[i + len(takers)]]
                self.total_reward += reward
 
 
        # 更新位置
        for taker in takers:
            if taker.reposition_target == 1:
                # print('taker repository了')
                continue

            if len (taker.order_list) > 1:
                
                # 接驾时间
                
                pickup_time = Config.unit_driving_time * \
                    self.network.get_path(taker.order_list[1].O_location, taker.location)
              
                self.taker_pickup_time.append(pickup_time)
                
                # 派送时间
                init_distance =  self.network.get_path(taker.order_list[0].O_location, taker.order_list[0].D_location) + \
                    self.network.get_path(taker.order_list[1].O_location, taker.order_list[1].D_location)

                actual_distance =  self.network.get_path(taker.order_list[1].O_location, taker.order_list[0].D_location) + \
                    self.network.get_path(taker.order_list[0].D_location, taker.order_list[1].D_location)

                dispatching_time = Config.unit_driving_time * actual_distance
                self.dispatch_time.append(dispatching_time)

                # 计算对于乘客来说，等待时间的增加
                waiting_time = self.time - taker.order_list[1].begin_time_stamp
                self.waiting_time.append(waiting_time)

                # 计算对于seeker来说，拼车绕行了多少
                self.extra_distance.append(actual_distance - init_distance)
                # 计算平台收益
                self.platflorm_income.append(Config.unit_distance_value/1000 *(actual_distance) )

                # 计算对于司机来说，拼车节省了多少距离
                init_distance = init_distance + \
                    self.network.get_path(taker.origin_location, taker.order_list[0].O_location) +\
                    self.network.get_path(taker.order_list[0].D_location, taker.order_list[1].O_location)
                actual_distance = actual_distance + \
                    self.network.get_path(taker.origin_location, taker.order_list[0].O_location)

                self.saved_distance.append(init_distance - actual_distance)


                # 计算拼车距离
                self.shared_distance.append(\
                    self.network.get_path(taker.order_list[1].O_location, taker.order_list[0].D_location))

                # 完成该拼车过程所花费的时间
                time_consume =  pickup_time + dispatching_time
                # 更新智能体可以采取动作的时间
                taker.activate_time += time_consume
                # print('拼车完成，activate_time:{}'.format(taker.activate_time - self.time))
                # 更新智能体的位置
                taker.location = taker.order_list[1].D_location
                # 完成订单
                taker.order_list = []
                taker.target = 0 # 变成vehicle
            else:
                # 派送时间
                dispatching_time = Config.unit_driving_time * \
                   (self.network.get_path(taker.order_list[0].O_location, taker.order_list[0].D_location)) 
                if self.time >= taker.activate_time + dispatching_time:
                    # 全程没拼到车，单独走到了终点
                    self.taker_pickup_time.append(Config.unit_driving_time * \
                   (self.network.get_path(taker.order_list[0].O_location, taker.location)) ) 
                    self.dispatch_time.append(dispatching_time)
                    self.platflorm_income.append(Config.unit_distance_value/1000 * \
                        self.network.get_path(taker.order_list[0].O_location, taker.order_list[0].D_location) )
                    self.shared_distance.append(0)

                    # 更新智能体可以采取动作的时间
                    taker.activate_time += dispatching_time
                    # print('没有拼到车，activate_time:{}'.format(taker.activate_time - self.time))
                    # 更新智能体的位置
                    taker.location = taker.order_list[0].D_location
                    # 完成订单
                    taker.order_list = []
                    taker.target = 0 # 变成vehicle
 
 
        for vehicle in vehicles:
            if vehicle.reposition_target == 1:
                # print('vehicle repository了')
                continue

            vehicle.target = 1 # 变成taker
            pickup_time = Config.unit_driving_time * self.network.get_path  \
                 (vehicle.location, vehicle.order_list[0].O_location)
            vehicle.origin_location = vehicle.location

            vehicle.location = vehicle.order_list[0].O_location
            self.taker_pickup_time.append(pickup_time)
            vehicle.activate_time += pickup_time

        return self.total_reward
 

    def calTakersWeights(self, taker, seeker,  optimazition_target, matching_condition):
        if optimazition_target == 'platform_income': 
            dispatch_distance = self.network.get_path(seeker.O_location, seeker.D_location)
            pick_up_distance = self.network.get_path(seeker.O_location, taker.location)
            if matching_condition and (pick_up_distance > self.pickup_distance_threshold \
                or dispatch_distance - pick_up_distance < 0):
                # print('taker pick_up_distance not pass', pick_up_distance)
                return 0
            else:
                # print('taker pick_up_distance', pick_up_distance,'dispatch_distance',dispatch_distance)
                reward = Config.unit_distance_value/1000 * dispatch_distance 
                value = self.critic.value_net( taker.location, self.time_slot)
                next_value = self.critic.value_net(seeker.D_location, self.time_slot)
                return reward + next_value - value

        else: # expected shared distance
            pick_up_distance = self.network.get_path(seeker.O_location, taker.location)
            shared_distance = self.network.get_path(seeker.O_location, taker.order_list[0].D_location)
            extra_distance = self.network.get_path(seeker.O_location, taker.order_list[0].D_location) + \
                self.network.get_path(taker.order_list[0].D_location,seeker.D_location) - \
                self.network.get_path(seeker.O_location, seeker.D_location)

            if  matching_condition and (pick_up_distance > self.pickup_distance_threshold or \
                extra_distance > self.extra_distance_threshold) :
                # print('extra_distance not pass', extra_distance)
                return 0
            else:
                reward = shared_distance
                value = self.critic.value_net( taker.location, self.time_slot)
                next_value = self.critic.value_net(seeker.D_location, self.time_slot)
                return reward + next_value - value


    def calVehiclesWeights(self, vehicle, seeker,  optimazition_target, matching_condition):
        if optimazition_target == 'platform_income':
            dispatch_distance = self.network.get_path(seeker.O_location, seeker.D_location)
            pick_up_distance = self.network.get_path(seeker.O_location, vehicle.location)
            if matching_condition and (pick_up_distance > Config.pickup_distance_threshold \
                or dispatch_distance - pick_up_distance < 0):
                # print('vehicle pick_up_distance not pass', pick_up_distance)
                return 0
            else:
                reward = Config.unit_distance_value/1000 *(dispatch_distance - pick_up_distance) 
                value = self.critic.value_net( vehicle.location, self.time_slot)
                next_value = self.critic.value_net(seeker.D_location, self.time_slot)
                return reward + next_value - value               

        else: # expected shared distance
            pick_up_distance = self.network.get_path(seeker.O_location, vehicle.location)
            if matching_condition and pick_up_distance > Config.pickup_distance_threshold:
                return 0
            else:
                reward = seeker.es
                value = self.critic.value_net( vehicle.location, self.time_slot)
                next_value = self.critic.value_net(seeker.D_location, self.time_slot)
                return reward + next_value - value
            


    def save_metric(self, path = "output/system_metric.pkl"):
        dic = {}
        dic['taker_pickup_time'] = self.taker_pickup_time
        dic['extra_distance'] = self.extra_distance
        dic['saved_distance'] = self.saved_distance
        dic['waiting_time'] = self.waiting_time
        self.success_rate = len(self.his_order)/ len(self.order_list)
        print('his_order{},all_order{}'.format(len(self.his_order),len(self.order_list)))
        dic['dispatch_time'] =self.dispatch_time
        dic['platflorm_income'] = self.platflorm_income
        dic['shared_distance'] = self.shared_distance
        dic['success_rate'] = self.success_rate


        import pickle

        with open(path, "wb") as tf:
            pickle.dump(dic,tf)
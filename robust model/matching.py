
import time
import pandas as pd
import numpy as np
import Network as net
import Seeker
import random
from common import KM_method
import Vehicle
import os
import pulp

import gurobipy as gp
from gurobipy import GRB

class matching():

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.date = cfg.date
        self.order_list = pd.read_csv(self.cfg.order_file).sample(
            frac=cfg.demand_ratio, random_state=1)
        # print('order_list',len(self.order_list))
        self.vehicle_num = int(len(self.order_list) /  cfg.order_driver_ratio )
        self.order_list['beginTime_stamp'] = self.order_list['dwv_order_make_haikou_1.departure_time'].apply(
            lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))) # arrive_time,departure_time
        # departure_time
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
        self.vehicle_list = []
        self.visualization_df = pd.DataFrame(columns=[
                                             'vehicle_id', 'vehicle_type', 'departure_time', 'departure_time', 'O_location', 'D_location', 'p1_id', 'p2_id'])
        self.takers = []
        self.current_seekers = []  # 存储需要匹配的乘客
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
            seeker.waitingtime_threshold = max(30, np.random.normal(self.cfg.delay_time_threshold, 10))
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
                
            self.res['waitingTime'] = np.mean(self.waitingtime)
            self.res['traveltime'] = np.mean(self.traveltime)
            self.res['detour_distance'] = np.mean(self.detour_distance)

            # for vehicle
            self.res['pickup_time'] = np.mean(self.pickup_time)
            self.res['shared_distance'] = np.mean(self.shared_distance)
            self.res['total_ride_distance'] = np.sum(
                self.total_travel_distance)
            self.res['avg_distance_saving']= np.mean(self.saved_travel_distance)
            self.res['saved_ride_distance'] = np.sum(
                self.saved_travel_distance)

            self.res['platform_income'] = np.sum(self.platform_income)
            # self.res['avg_platform_income']= np.sum(self.platform_income) / len(self.his_order)
            self.res['avg_platform_income']= np.mean(self.platform_income) 
            self.res['response_rate'] = len(list(set(self.his_order))) / len(self.order_list)
            self.res['carpool_rate'] = len(self.carpool_order) / len(self.his_order)
            self.res['ride_distance_error'] = self.ride_distance_error
            self.res['shared_distance_error'] = self.shared_distance_error
            self.res['relative_ride_distance_error'] = self.relative_ride_distance_error
            self.res['relative_shared_distance_error'] = self.relative_shared_distance_error

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
            self.activate_drivers = 0
            for vehicle in self.vehicle_list:
                # print('vehicle.activate_time',vehicle.activate_time)
                # print('vehicle.state',vehicle.state)
                vehicle.is_activate(time_)
                vehicle.reset_reposition()

                if vehicle.state == 1:  # 能执行动作
                    # print('id{},vehicle.target{}'.format(vehicle.id, vehicle.target))
                    self.activate_drivers += 1
                    if vehicle.target == 0:
                        vehicles.append(vehicle)
                    else:
                        # print('id{},vehicle.target{}'.format(vehicle.id, vehicle.target))
                        takers.append(vehicle)
            # print('len(vehicles)',len(vehicles),'len(takers)', len(takers))
            start = time.time()
            reward = self.batch_matching(takers, vehicles, seekers)
            end = time.time()
            if self.cfg.progress_target:
                print('匹配用时{},time{},vehicles{},takers{},seekers{}'.format(end - start, self.time_slot, len(vehicles),len(takers) ,len(seekers)))   
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
        matrix = np.ones((dim, dim)) * -8000
        d = np.ones((dim, dim))
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
                        d[row,column] = matrix[row, column] * 0.3
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
                        d[row,column] = matrix[row, column] * 0.3
                        end = time.time()
                        # print('计算Vehicle权重时间', end - start)
                    else:
                        continue

        # 计算司机选择调度的权重
        for row in range(len(takers) + len(vehicles)):
            for column in range( len(seekers), dim):
                matrix[row, column] = - 4000

        # 计算乘客选择等待的权重
        for column in range(len(seekers)):
            for row in range(len(takers) + len(vehicles), row_nums):
                matrix[row, column] = \
                    self.calSeekerWaitingWeights(seekers[column],
                                                 optimazition_target=self.optimazition_target)
                d[row,column] = matrix[row, column] * 0.3

        # 匹配
        if demand == 0 or supply == 0:
            self.remain_seekers = []
            for seeker in seekers:
                if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < self.cfg.delay_time_threshold:
                    seeker.set_delay(self.time)
                    self.remain_seekers.append(seeker)
            return 0


        start = time.time()
        # Create a new model
        model = gp.Model("Optimization Problem")
        # model.setParam('LogFile', 'gurobi.log')
        model.setParam(GRB.Param.OutputFlag, 0)  # 禁止输出求解信息
        # Define the variables
        V, P= range(dim), range(dim)
        x = model.addVars(V, P, vtype=GRB.BINARY)
        y = model.addVars(V, P, vtype=GRB.BINARY)
        z = model.addVar(lb=0)
        q = model.addVars(V, P, lb=0)
        Gamma = self.cfg.gamma * dim
        # Define the objective function
        objective = gp.quicksum(-matrix[v][p] * x[v, p] + z * Gamma + q[v, p] for p in P for v in V)
        # objective = gp.quicksum(-matrix[v][p] * x[v, p] for p in P for v in V) + z * Gamma + gp.quicksum(q[v, p] for p in P for v in V)
        model.setObjective(objective, GRB.MINIMIZE)

        # Define the constraints
        for p in P:
            model.addConstr(gp.quicksum(x[v, p] for v in V) <= 1)

        for v in V:
            model.addConstr(gp.quicksum(x[v, p] for p in P) <= 1)

        for p in P:
            for v in V:
                model.addConstr(z + q[v, p] >= d[v][p] * y[v, p])
                model.addConstr(-x[v, p] <= y[v, p])
                model.addConstr(y[v, p] <= x[v, p])

        # Optimize the model
        model.optimize()

        # Check the status of the solution
        status = model.status
        # print("Status:", status)

        # Get the optimal variable values
        optimal_values = {(v, p): x[v, p].x for v in V for p in P}
        end = time.time()

        # print('matching time:',end-start)
        # Print the optimal variable values for x
        # for v in range(dim):
        #     for p in range(dim):
        #         if optimal_values[(v, p)] == 1:
        #             print(f"x[{v}][{p}] = {optimal_values[(v, p)]}")

        for v in range(len(takers)):
            takers[v].reposition_target = 1
            #  第i个vehicle响应第res[1][i]个订单
            for p in range(dim):
                if optimal_values[v+len(takers),p] == 1:
                    if p >= len(seekers):
                    # 接到乘客
                    # 接到了虚拟订单，taker应该进入下一个匹配池
                        takers[v].reposition_target = 1
                    # print('taker{}拼车失败，进入匹配池'.format(
                    #     takers[i].id))
                    else:
                        takers[v].order_list.append(seekers[p])
                        takers[v].reposition_target = 0
                        # print('taker{}拼车成功，权重为{}'.format(
                        #     takers[i].id, matrix[i, res[i]]))
                        # 记录seeker等待时间
                        seekers[p].set_waitingtime(
                            self.time - seekers[p].begin_time_stamp)
                        seekers[p].response_target = 1

        for v in range(len(vehicles)):
            vehicles[v].reposition_target = 1
            #  第i个vehicle响应第res[1][i]个订单
            for p in range(dim):
                if optimal_values[v+len(takers),p] == 1:
                    if p >= len(seekers):
                        # 接到了虚拟订单
                        vehicles[v].reposition_target = 1
                        if not vehicles[v].path:
                            repostion_location = random.choice(self.locations)
                            distance, vehicles[v].path, vehicles[v].path_length = self.network.get_path(
                                    vehicles[v].location, repostion_location)
                    else:
                        # print('vehicle id{},order id{}'.format(vehicles[i].id, seekers[res[i + len(takers)]].id))
                        vehicles[v].order_list.append(seekers[p])
                        vehicles[v].reposition_target = 0
                        # print('vehicles{}拼车成功，权重为{}'.
                        #       format(vehicles[i].id, matrix[i + len(takers), res[i + len(takers)]]))
                        # 记录seeker等待时间
                        seekers[p].set_waitingtime(self.time - seekers[p].begin_time_stamp)
                        seekers[p].response_target = 1

        # print('匹配时间{},匹配成功{},匹配失败{},takers{},vehicles{},demand{},time{}'.
        #       format(end-start, successed, failed, len(takers), len(vehicles), len(seekers), self.time_slot))
        start = time.time()


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
                        self.cfg.unit_distance_cost/1000 * ( taker.order_list[0].shortest_distance + taker.p0_pickup_distance)
                    )
                    # print('order.value{},driver_distance{},income{}:'.format(taker.order_list[0].value,taker.order_list[0].shortest_distance + taker.p0_pickup_distance,self.platform_income[-1]))
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
                        self.platform_income.append(self.cfg.discount_factor * taker.order_list[0].value -self.cfg.unit_distance_cost/1000 * ( taker.order_list[0].shortest_distance + taker.p0_pickup_distance))
                        # print('order.value{},driver_distance{},income{}:'.format(taker.order_list[0].value,taker.order_list[0].shortest_distance + taker.p0_pickup_distance,self.platform_income[-1]))
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

                # print('order.value{},driver_distance{},income:{}'.format(self.cfg.discount_factor * (taker.order_list[0].value + taker.order_list[1].value),\
                #     (p0_invehicle + p1_invehicle -distance[0] + taker.p0_pickup_distance),\
                #         self.cfg.discount_factor * (taker.order_list[0].value + taker.order_list[1].value) -
                #     self.cfg.unit_distance_cost/1000 * (p0_invehicle + p1_invehicle -distance[0] + taker.p0_pickup_distance)))
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
                # print('matched',vehicle.id)
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

        return step_reward


    def calTakersWeights(self, taker, seeker,  optimazition_target, matching_condition):
        # expected shared distance
        pick_up_distance = min(self.get_path(
            seeker.O_location, taker.order_list[0].O_location), self.get_path(
            taker.location, taker.order_list[0].O_location))
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

        distance_saving = seeker.shortest_distance + taker.order_list[0].shortest_distance - (p0_invehicle + p1_invehicle - shared_distance)

        if optimazition_target == 'platform_income':
            if matching_condition and (pick_up_distance > self.pickup_distance_threshold or
                                    p0_detour > self.detour_distance_threshold or
                                    p1_detour > self.detour_distance_threshold):
                # print('detour_distance not pass', detour_distance)
                return self.cfg.dead_value
            else:
                profit = self.cfg.discount_factor * (seeker.value + taker.order_list[0].value) - \
                    self.cfg.unit_distance_cost/1000 *(p0_invehicle + p1_invehicle - shared_distance+ taker.p0_pickup_distance ) # 
                # print('balanced taker reward',profit)
                return profit* seeker.delay
        
        elif optimazition_target == 'combination':  
            profit = self.cfg.discount_factor * (seeker.value + taker.order_list[0].value) - \
                self.cfg.unit_distance_cost/1000 *(p0_invehicle + p1_invehicle - shared_distance + taker.p0_pickup_distance)
            
            return self.cfg.beta * distance_saving + (1-self.cfg.beta) * profit 
        else:
            if matching_condition and (pick_up_distance > self.pickup_distance_threshold or
                                    p0_detour > self.detour_distance_threshold or
                                    p1_detour > self.detour_distance_threshold or distance_saving<0):
                # print('detour_distance not pass', detour_distance)
                return self.cfg.dead_value
            else:

                if (distance_saving + pick_up_distance) == 0:
                    return 0
                reward = distance_saving * (distance_saving/ (distance_saving + pick_up_distance)) * seeker.delay

                return reward


    def calVehiclesWeights(self, vehicle, seeker,  optimazition_target, matching_condition):
        pick_up_distance = self.get_path(
            seeker.O_location, vehicle.location)
        
        if optimazition_target == 'platform_income':
            if matching_condition and (pick_up_distance > self.pickup_distance_threshold):
                # print('detour_distance not pass', detour_distance)
                return self.cfg.dead_value
            else:
                profit = self.cfg.discount_factor * (seeker.value + self.cfg.unit_distance_value / 1000 * seeker.expected_pooling_passenger_ride_distance_for_taker )-\
                self.cfg.unit_distance_cost/1000 *(pick_up_distance + seeker.expected_pooling_driver_ride_distance_for_taker )

                # print('balanced vacant reward',profit)
                return 0.7 * profit* seeker.delay

        
        elif optimazition_target == 'combination':
            distance_saving = seeker.esdt * (seeker.esdt / (seeker.esdt + pick_up_distance )) * seeker.delay
            profit = seeker.value + self.cfg.unit_distance_value / 1000 * seeker.expected_pooling_passenger_ride_distance_for_taker -\
                self.cfg.unit_distance_cost/1000 *(pick_up_distance + seeker.expected_pooling_driver_ride_distance_for_taker)
            return self.cfg.beta * distance_saving + (1-self.cfg.beta) * profit 
        
        else:
            if matching_condition and (pick_up_distance > self.cfg.pickup_distance_threshold ):
                return self.cfg.dead_value
            else:
                reward = seeker.esdt * (seeker.esdt / (seeker.esdt + pick_up_distance )) * seeker.delay

                return reward


    # 计算乘客选择等待的权重
    def calSeekerWaitingWeights(self, seeker,  optimazition_target):
        if optimazition_target == 'platform_income':
            profit = seeker.value + self.cfg.unit_distance_value / 1000 * seeker.expected_pooling_passenger_ride_distance_for_seeker -\
                self.cfg.unit_distance_cost/1000 *(seeker.expected_pooling_driver_ride_distance_for_seeker)

            reward = profit * (1 - (1- self.cfg.rw) ** (max(0,self.cfg.delay_time_threshold / 10 - seeker.k ))) /(seeker.k + 1) #   # np.exp(seeker.k)
            # print('waitiing reward',reward)
            return reward
        
        elif optimazition_target == 'combination':
            distance_saving = seeker.esds
            profit = seeker.value + self.cfg.unit_distance_value / 1000 * seeker.expected_pooling_passenger_ride_distance_for_seeker -\
                self.cfg.unit_distance_cost/1000 *(seeker.expected_pooling_driver_ride_distance_for_seeker)
            reward = (self.cfg.beta * distance_saving + (1-self.cfg.beta) * profit ) \
                * (1 - (1- self.cfg.rw) ** ((max(0,self.cfg.delay_time_threshold / 10 - seeker.k )) )) 
            # gamma = 11.67
            # reward =  (self.cfg.beta * distance_saving + (1-self.cfg.beta) * profit )  - gamma * (self.time - seeker.begin_time_stamp) /10
            return reward 

        else:  # expected shared distance
            gamma = 11.67
            # gamma = 3
            if seeker.esds < 0:
                return seeker.esds
            reward = seeker.esds * (1 - (1- self.cfg.rw) ** (max(0,self.cfg.delay_time_threshold / 10 - seeker.k ))) /  (seeker.k + 1)# np.exp(seeker.k) (seeker.k + 1)
            if seeker.waitingWeight == 0:
                seeker.waitingWeight = reward
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

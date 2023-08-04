
import sys, os
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前路径
parent_path=os.path.dirname(curr_path) # 父路径，这里就是我们的项目路径
sys.path.append(parent_path) # 由于需要引用项目路径下的其他模块比如envs，所以需要添加路径到sys.path

import random 

import Config
import Network


class utils():

    Config = Config.Config()
    Network = Network.Network()

    def __init__(self):
        print('import utils')


    def show(self):
        print('show utils')

    # 返回指定时间窗的订单
    def get_time_interval_orders(self,  orders):
        # 把这些订单统计到每个rectangle
        self.arrange_orders_to_rectangles(orders)
        return orders



    def get_sharing_order(self, location, current_time_orders):
        neighbors_loc = self.Network.get_neighbors(location)

        for location in neighbors_loc:
            orders = current_time_orders[current_time_orders['O_location'] == location.name]
            if len(orders) > 0 :
                # 从这个网格内随便接一个拼车单
                order = orders.sample(1, replace = False, axis = 0)
                return order
        # 如果附近没有订单，就从当前时间窗内的订单中抽一个给司机
        order = current_time_orders.sample(1, replace = False, axis = 0)
        return order


    def get_general_order(self, location, current_time_orders):

        neighbors_loc = self.Network.get_neighbors(location)
        
        for location in neighbors_loc:
            orders = current_time_orders[current_time_orders['O_location'] == location.name]
            if len(orders) > 0 :
                # 从这个网格内随便接一个拼车单
                order = orders.sample(1, replace = False, axis = 0)

                return order
        # 如果附近没有订单，就从当前时间窗内的订单中抽一个给司机
        order = current_time_orders.sample(1, replace = False, axis = 0)
        return order





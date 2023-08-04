
import Config
import time



class Vehicle:
    

    def __init__(self, driver_id, location,cfg):
        self.id = driver_id
        self.passengers = 0 # 0 没有乘客，1 可接一个拼车乘客，2 不可接乘客
        self.order_list = [] # 当前正在响应的订单

        self.reward = 0
        self.target = 0
        self.reposition_target = 0 # 没有订单可接，进入下一次匹配


        self.location = location # 当前智能体的位置
        self.origin_location = location # 拼车前的位置
        self.destination = 0 # 当前智能体的目的地
        self.state = 0 # 能否执行动作
        self.value = 0
        self.target_value = 0
        self.activate_time  = time.mktime(time.strptime(cfg.date + cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))


    # 判断司机是否能执行动作
    def is_activate(self, time):
        if time > self.activate_time:
            self.state = 1
        else:
            self.state = 0

    # 重置reposition 指令
    def reset_reposition(self):
        self.reposition_target = 0

        
    # 重置
    def reset(self):
        cfg = Config.Config()
        self.passengers = 0 # 0 没有乘客，1 可接一个拼车乘客，2 不可接乘客
        self.order_list = [] # 当前正在响应的订单
        self.his_order_list = [] # 响应过的所有订单
        self.reward = 0

        self.path = [] # 当前的行驶路径
        self.state = 0 # 能否执行动作
        self.activate_time  = time.mktime(time.strptime(cfg.date + cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))


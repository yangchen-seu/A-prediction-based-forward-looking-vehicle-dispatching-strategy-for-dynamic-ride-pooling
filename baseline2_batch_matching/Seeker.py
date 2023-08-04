'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-07-03 09:04:17
LastEditors: yangchen-seu 58383528+yangchen-seu@users.noreply.github.com
LastEditTime: 2023-01-10 01:16:55
FilePath: /matching/reinforcement learning/Seeker.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import time
import math

class Seeker():

    def __init__(self, row ) -> None:

        self.id = row['dwv_order_make_haikou_1.order_id']
        self.begin_time = row['dwv_order_make_haikou_1.departure_time']
        # self.begin_time = row['dwv_order_make_haikou_1.arrive_time']
        self.O_location = row.O_location
        self.D_location = row.D_location
        self.begin_time_stamp = time.mktime(time.strptime\
            (self.begin_time, "%Y-%m-%d %H:%M:%S"))
        self.matching_prob = row.matching_prob
        self.rs = row.ride_distance
        self.detour_distance = row.detour_distance
        self.es = row.shared_distance
        self.rst = row.ride_distance_for_taker
        self.detour_distance_for_taker = row.detour_distance_for_taker
        self.est = row.shared_distance_for_taker
        self.service_target = 0
        self.detour = 0
        self.shortest_distance = 0
        self.traveltime = 0
        self.waitingtime = 0
        self.delay = 1
        self.response_target = 0
        self.k = 1
        self.value = 0
        self.shared_distance = 0
        self.ride_distance = 0
        self.esds = row.saved_distance_for_seeker * 2
        self.esdt = row.saved_distance_for_taker * 2

        self.waitingWeight = 0
        self.soloWeight = 0
        self.matchingWeight = 0
        self.random_seed = 1
        self.waitingtime_threshold = 90


    def show(self):
        print('self.id', self.id)
        print('self.begin_time',  self.begin_time)
        print('self.O_location', self.O_location)
        print('self.D_location',  self.D_location)
        print('self.begin_time_stamp', self.begin_time_stamp)
        print('self.matching_prob',  self.matching_prob)
        print('self.rs ', self.rs)
        print('self.detour_distance ', self.detour_distance)
        print('self.es ', self.es)
        print('self.rst ', self.rst)
        print('self.detour_distance_for_taker ', self.detour_distance_for_taker)
        print('self.est ', self.est)
        print('self.detour', self.detour)
        print('self.shortest_distance', self.shortest_distance)
        print('self.traveltime ', self.traveltime )
        print('self.waitingtime ', self.waitingtime )
        print('self.delay ', self.delay )
        print('self.k ', self.k )
        print('self.value ', self.value )
        print('self.shared_distance ', self.shared_distance)
        print('self.ride_distance ', self.ride_distance)    

    def set_delay(self, time):
        self.k = math.floor((time - self.begin_time_stamp) / 10 )
        self.delay = 1.1 ** self.k

    def set_value(self,value):
        self.value = value
        
    def set_shortest_path(self,distance):
        self.shortest_distance = distance

    def set_waitingtime(self, waitingtime):
        self.waitingtime = waitingtime

    def set_traveltime(self,traveltime):
        self.traveltime = traveltime

    def set_detour(self,detour):
        self.detour = detour

    def cal_expected_ride_distance_for_wait(self, gamma):
        self.shared_distance  =self.shared_distance * gamma

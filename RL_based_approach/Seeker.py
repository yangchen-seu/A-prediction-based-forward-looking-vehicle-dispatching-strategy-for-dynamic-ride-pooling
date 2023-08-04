
import time

class Seeker():

    def __init__(self,index, row ) -> None:

        self.id = index
        self.begin_time = row['dwv_order_make_haikou_1.arrive_time']
        self.O_location = row.O_location
        self.D_location = row.D_location
        self.begin_time_stamp = time.mktime(time.strptime\
            (self.begin_time, "%Y-%m-%d %H:%M:%S"))
        self.matching_prob = row.matching_prob
        self.ls = row.ride_distance
        self.detour_distance = row.detour_distance
        self.es = row.shared_distance
        self.lst = row.ride_distance_for_taker
        self.detour_distance_for_taker = row.detour_distance_for_taker
        self.est = row.shared_distance_for_taker


    def cal_expected_ride_distance_for_wait(self, gamma):
        self.shared_distance  =self.shared_distance * gamma

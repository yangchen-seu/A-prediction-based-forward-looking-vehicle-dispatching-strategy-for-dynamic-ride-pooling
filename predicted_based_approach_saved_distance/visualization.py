'''
Author: yangchen-seu 58383528+yangchen-seu@users.noreply.github.com
Date: 2023-01-06 14:54:38
LastEditors: yangchen-seu 58383528+yangchen-seu@users.noreply.github.com
LastEditTime: 2023-01-06 17:16:33
FilePath: \forward-looking-ridepooling-matching-algorithm\code\matching\predicted_based_approach_saved_distance\visualization.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# In[]
import pandas as pd

drivers = pd.read_csv('./output/visualization_df.csv')
passengers = pd.read_csv('./input/order.csv')
nodes = pd.read_csv('./input/node.csv')

drivers = pd.merge(left = drivers, right= nodes[['node_id','x_coord','y_coord']], left_on= 'O_location', right_on= 'node_id')
drivers.columns = ['Unnamed: 0', 'vehicle_id', 'vehicle_type', 'depart_timestamp','arrival_timestamp', 'O_location',
       'D_location', 'p1_id', 'p2_id', 'node_id', 'O_x_coord', 'O_y_coord']

drivers = pd.merge(left = drivers, right= nodes[['node_id','x_coord','y_coord']], left_on= 'D_location', right_on= 'node_id')
drivers.columns = ['Unnamed: 0', 'vehicle_id', 'vehicle_type', 'depart_timestamp','arrival_timestamp', 'O_location',
       'D_location', 'p1_id', 'p2_id', 'O_node_id', 'O_x_coord', 'O_y_coord','D_node_id', 'D_x_coord', 'D_y_coord']

drivers['depart_time'] = pd.to_datetime(drivers['depart_timestamp'],unit='s')
drivers['arrival_time'] = pd.to_datetime(drivers['arrival_timestamp'],unit='s')
drivers[['vehicle_id', 'vehicle_type', 'depart_time', 'depart_timestamp', 'arrival_time','arrival_timestamp','O_location',
       'D_location', 'p1_id', 'p2_id',  'O_x_coord', 'O_y_coord', 'D_x_coord', 'D_y_coord']].to_csv('output/drivers.csv')

#In[]
passengers = pd.merge(left = passengers, right= nodes[['node_id','x_coord','y_coord']], left_on= 'O_location', right_on= 'node_id')
print(passengers.columns)
passengers.columns = ['Unnamed: 0', 'order_id',
       'arrive_time', 'O_location', 'D_location',
       'OD_id', 'matching_prob', 'ride_distance', 'detour_distance',
       'shared_distance', 'ride_distance_for_taker',
       'detour_distance_for_taker', 'shared_distance_for_taker',
       'ride_distance_for_seeker', 'detour_distance_for_seeker',
       'shared_distance_for_seeker', 'saved_distance_for_seeker',
       'saved_distance_for_taker', 'origin_id', 'destination_id', 'lambda',
       'node_id', 'x_coord', 'y_coord']

# In[]
import time
import pandas as pd


# 使用 to_datetime 函数将其转换为时间戳
passengers[[ 'order_id',
       'arrive_time', 'O_location', 'D_location',
       'node_id', 'x_coord', 'y_coord']].to_csv('output/passengers.csv')
# In[] 分析
# drivers['vehicle_id'].unique()
pd.set_option('display.float_format',lambda x : '%.2f' % x)

tmp = drivers[drivers.vehicle_id == 0]
tmp = tmp.sort_values(by = ['depart_time'])
tmp
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:30:21 2022

@author: May
"""

# In[]
from tracemalloc import start
import pandas as pd
link = pd.read_csv('link.csv')
inverse = pd.DataFrame()
inverse['from_node_id'] = link['to_node_id']
inverse['to_node_id'] = link['from_node_id']
inverse['length'] = link['length']

res = pd.concat([link, inverse] , axis = 0,sort=True)
print(len(link))
print(len(res))
print(len(inverse))
res.tail()
res.to_csv('links.csv')
# In[]
import pandas as pd
orders = pd.read_csv('../orders.csv', sep= '\t')
node = pd.read_csv('../node.csv')
print(orders.columns)
orders.head()
# In[]
def returnKey(dic, value):
       for i,j in dic.items():
              if j==value:
                     return i

# 
# orders = pd.read_csv('../orders.csv', sep= '\t')
# print(len(orders))
# orders = orders[orders['dwv_order_make_haikou_1.day'] == 1]

# print(len(orders))
i = 0
for index, row in tmp.iterrows():
       # print(index)
       if i%1000 == 0:
              print(i)
       i += 1
       start_distance = {}
       end_distance = {}
       for index1,row1 in node.iterrows():
              start_distance[row1['node_id']] = \
            abs(row['dwv_order_make_haikou_1.starting_lng']  -   row1['x_coord']) + \
            abs(row['dwv_order_make_haikou_1.starting_lat']  -   row1['y_coord'])
              end_distance[row1['node_id']] = \
            abs(row['dwv_order_make_haikou_1.dest_lng']  -   row1['x_coord']) + \
            abs(row['dwv_order_make_haikou_1.dest_lat']  -   row1['y_coord'])
       # print(start_distance)
       # print(returnKey(start_distance, min(start_distance.values())))
       tmp.loc[index,'O_loction'] = returnKey(start_distance, min(start_distance.values()))
       tmp.loc[index,'D_loction'] = returnKey(end_distance, min(end_distance.values()))
       
tmp.to_csv('orders_tmp.csv')
# In[]
Index(['dwv_order_make_haikou_1.order_id',
       'dwv_order_make_haikou_1.product_id', 'dwv_order_make_haikou_1.city_id',
       'dwv_order_make_haikou_1.district', 'dwv_order_make_haikou_1.county',
       'dwv_order_make_haikou_1.type', 'dwv_order_make_haikou_1.combo_type',
       'dwv_order_make_haikou_1.traffic_type',
       'dwv_order_make_haikou_1.passenger_count',
       'dwv_order_make_haikou_1.driver_product_id',
       'dwv_order_make_haikou_1.start_dest_distance',
       'dwv_order_make_haikou_1.arrive_time',
       'dwv_order_make_haikou_1.departure_time',
       'dwv_order_make_haikou_1.pre_total_fee',
       'dwv_order_make_haikou_1.normal_time',
       'dwv_order_make_haikou_1.bubble_trace_id',
       'dwv_order_make_haikou_1.product_1level',
       'dwv_order_make_haikou_1.dest_lng', 'dwv_order_make_haikou_1.dest_lat',
       'dwv_order_make_haikou_1.starting_lng',
       'dwv_order_make_haikou_1.starting_lat', 'dwv_order_make_haikou_1.year',
       'dwv_order_make_haikou_1.month', 'dwv_order_make_haikou_1.day'],
      dtype='object')
#In[]
tmp
# In[]

import pandas as pd
orders = pd.read_csv('orders.csv')
OD = pd.read_csv('OD.csv')
predict_result = pd.read_csv('predict_result.csv')

# In[]
print(orders[['O_location','D_location']].head())
print(predict_result.columns)

# In[]
predict_result = predict_result.merge(OD, left_on = 'OD_id', right_on = 'OD_id')
print(predict_result.columns)
print(orders.columns)
predict_result['destination_id'] = predict_result['destination_id'].astype(int)
predict_result['origin_id'] = predict_result['origin_id'].astype(int)


# In[]
res = pd.merge(orders, predict_result, left_on = ['O_location','D_location'], \
               right_on = ['origin_id','destination_id'])
res.head()
# In[]
#res.to_csv('order.csv')
print(res.columns)

res[[ 'dwv_order_make_haikou_1.order_id', 'dwv_order_make_haikou_1.arrive_time', 
       'O_location', 'D_location', 'OD_id', 'matching_prob', 'ride_distance',
       'detour_distance', 'shared_distance', 'ride_distance_for_taker',
       'detour_distance_for_taker', 'shared_distance_for_taker',
       'ride_distance_for_seeker', 'detour_distance_for_seeker',
       'shared_distance_for_seeker',
       'origin_id', 'destination_id', 'lambda']].to_csv('order.csv')
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:30:21 2022

@author: May
"""

import pandas as pd
import os

predict_results = os.listdir('../predict_results/')
orders = os.listdir('../orders/')
ODs = os.listdir('../ODs/')
for i in range(len(ODs)):
       print(ODs[i])
       order  = pd.read_csv('../orders/'+ orders[i])
#        print(order.columns)
       order = order[['dwv_order_make_haikou_1.order_id',
       'dwv_order_make_haikou_1.departure_time', 'O_location', 'D_location']]
       OD = pd.read_csv('../ODs/'+ ODs[i])
       predict_result = pd.read_csv('../predict_results/'+ predict_results[i])
       predict_result = pd.merge(OD, predict_result,left_on = 'OD_id', right_on = 'OD_id')
       predict_result['destination_id'] = predict_result['destination_id'].astype(int)
       predict_result['origin_id'] = predict_result['origin_id'].astype(int)
#        order['O_location'] = order['O_loction']
#        order['D_location'] = order['D_loction']
       res = pd.merge(order, predict_result, left_on = ['O_location','D_location'], \
               right_on = ['origin_id','destination_id'])
#        print(res.columns)

       res[[ 'dwv_order_make_haikou_1.order_id', 'dwv_order_make_haikou_1.departure_time', 
       'O_location', 'D_location', 'OD_id', 'matching_prob', 'ride_distance',
       'detour_distance', 'shared_distance', 'ride_distance_for_taker',
       'detour_distance_for_taker', 'shared_distance_for_taker',
       'ride_distance_for_seeker', 'detour_distance_for_seeker',
       'shared_distance_for_seeker','saved_distance_for_seeker',
       'saved_distance_for_taker','seeker_actual_ride_distance',
       'taker_expect_ride_distance', 'driver_expect_ride_distance',
       'taker_expect_ride_distance_for_taker',
       'driver_expect_ride_distance_for_taker',
       'taker_expect_ride_distance_for_seeker',
       'driver_expect_ride_distance_for_seeker',
       'origin_id', 'destination_id', 'lambda']].to_csv('../matched_orders/'+orders[i])

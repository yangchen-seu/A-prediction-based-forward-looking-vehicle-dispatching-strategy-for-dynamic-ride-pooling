# In[]
import pandas as pd
import pickle
with open('results.pkl','rb') as tf:
    results = pickle.load(tf)
    print(results.keys())
    for key in results.keys():
        print('date:{},waitingTime{},detour_distance{},pickup_time{},shared_distance{},\
        total_ride_distance{},saved_ride_distance{},platform_income{},\
            response_rate{},carpool_rate{}'.format(key,results[key]['waitingTime'],\
                results[key]['detour_distance'],results[key]['pickup_time'],\
                    results[key]['shared_distance'],results[key]['total_ride_distance'],\
                        results[key]['saved_ride_distance'],results[key]['platform_income'],\
                            results[key]['response_rate'],results[key]['carpool_rate']))


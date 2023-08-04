import pickle
import numpy as np

with open("system_metric.pkl", 'rb') as f:

    res: dict = pickle.load(f)

print(res.keys())
print(np.mean(res['taker_pickup_time']))
print(np.mean(res['extra_distance']))
print(np.mean(res['saved_distance']))
print(np.mean(res['waiting_time']))
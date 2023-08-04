import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
files = os.listdir('input/orders/')
folder = 'output/'
dic = {}

waiting_time = []
detour_distance = []
pickup_time = []
shared_distance = []
total_ride_distance = []
saved_ride_distance = []
platform_income = []
response_rate = []
carpool_rate = []
x = []

for file in files:
    with open(folder + file.split('.')[0] + '/system_metric.pkl', 'rb') as f:
        data = pickle.load(f)
        # print(data.keys())
        x.append(file.split('.')[0][-2:])
        waiting_time.append(np.mean(data['waiting_time']))
        detour_distance.append(np.mean(data['detour_distance']))
        pickup_time.append(np.mean(data['pickup_time']))
        shared_distance.append(np.mean(data['shared_distance']))
        total_ride_distance.append(np.mean(data['total_ride_distance']))
        saved_ride_distance.append(np.mean(data['saved_distance']))
        platform_income.append(np.mean(data['platform_income']))
        response_rate.append(np.mean(data['response_rate']))
        carpool_rate.append(np.mean(data['carpool_rate']))
        if file.split('.')[0][-2:] == '01':
            print('waiting_time:',waiting_time,'detour_distance:',detour_distance,'pickup_time:',pickup_time,'total_ride_distance:',total_ride_distance,'platform_income:',platform_income,\
                'response_rate:',response_rate,'shared_distance:',shared_distance,'saved_distance:',saved_ride_distance,'carpool_rate:',carpool_rate)

# 绘图
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
def plot_metrics(  x,y,  metric_name):    
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.xlabel('days')
    plt.plot(x,y, label=metric_name)
    plt.legend()
    plt.savefig(
            'output/{}.png'.format(metric_name))

# plot_metrics(x,waiting_time,'waiting_time')
# plot_metrics(x,detour_distance,'detour_distance')
# plot_metrics(x,pickup_time,'pickup_time')
# plot_metrics(x,shared_distance,'shared_distance')
# plot_metrics(x,total_ride_distance,'total_ride_distance')
# plot_metrics(x,saved_ride_distance,'saved_ride_distance')
# plot_metrics(x,platform_income,'platform_income')
# plot_metrics(x,response_rate,'response_rate')
# plot_metrics(x,carpool_rate,'carpool_rate')

dic = {}
dic['waiting_time'] = waiting_time
dic['detour_distance'] = detour_distance
dic['pickup_time'] = pickup_time
dic['shared_distance'] = shared_distance
dic['total_ride_distance'] = total_ride_distance
dic['saved_ride_distance'] = saved_ride_distance
dic['platform_income'] = platform_income
dic['response_rate'] = response_rate
dic['carpool_rate'] = carpool_rate

df = pd.DataFrame(dic)
df.head()

 ####这句用于处理边界数字越界显示不全####
plt.ylim(0,len(df.corr()))

 ####解决保存图时坐标文字显示不全#######
plt.tight_layout()

sns.heatmap(df.corr())
plt.savefig('output/correlationship.jpg', dpi=100, bbox_inches='tight')
plt.show()
# print(waiting_time)

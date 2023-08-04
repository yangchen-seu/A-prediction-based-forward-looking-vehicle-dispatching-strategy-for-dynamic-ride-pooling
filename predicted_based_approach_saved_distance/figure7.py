# In[]
import pickle
import Seeker
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
node = pd.read_csv('input/node.csv')

res = {}
detour = 3000

path = "./output/expected_shared_distance/his_order.pkl"
fr = open(path, 'rb')
order = pickle.load(fr)


O_location = []
D_location = []
for key in order.keys():
    O_location.append(order[key].O_location)
    D_location.append(order[key].D_location)

O_lat = []
O_lon = []
for i in O_location:
    O_lat.append(node[node['node_id'] == i]['y_coord'].values[0])
    O_lon.append(node[node['node_id'] == i]['x_coord'].values[0])

D_lat = []
D_lon = []
for i in D_location:
    D_lat.append(node[node['node_id'] == i]['y_coord'].values[0])
    D_lon.append(node[node['node_id'] == i]['x_coord'].values[0])

data = {'O_location': O_location,
            'O_lat': O_lat,
            'O_lon': O_lon,
        'D_location': D_location,
            'D_lat': D_lat,
            'D_lon': D_lon,
            }
df = pd.DataFrame(data)

df.head()
plt.scatter(df['O_lon'], df['O_lat'], marker = '*', c='r')
plt.scatter(df['D_lon'], df['D_lat'], c='g')
plt.savefig(path+'OD_location.png')

df.to_csv('saved_distance_OD_distribution.csv')
# In[] 查看历史订单重复数据
import pickle
import Seeker
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
node = pd.read_csv('input/node.csv')
path = "./output/expected_shared_distance/his_order.pkl"
fr = open(path, 'rb')
order = pickle.load(fr)
id = []
O_location = []
D_location = []
for key in order.keys():
    id.append(order[key].id)
    O_location.append(order[key].O_location)
    D_location.append(order[key].D_location)

O_lat = []
O_lon = []
for i in O_location:
    O_lat.append(node[node['node_id'] == i]['y_coord'].values[0])
    O_lon.append(node[node['node_id'] == i]['x_coord'].values[0])

D_lat = []
D_lon = []
for i in D_location:
    D_lat.append(node[node['node_id'] == i]['y_coord'].values[0])
    D_lon.append(node[node['node_id'] == i]['x_coord'].values[0])

data = {'id':id,
    'O_location': O_location,
            'O_lat': O_lat,
            'O_lon': O_lon,
        'D_location': D_location,
            'D_lat': D_lat,
            'D_lon': D_lon,
            }
df = pd.DataFrame(data)

df.head()
# In[]
print(df)
print(df.duplicated(subset='id'))
print(len(df))
print(len(df['id'].unique()))



import Main

# 绘图
def plot_metrics(metric,  sensitive_param, metric_name, param_name ):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    plt.figure() # 创建一个图形实例，方便同时多画几个图
    plt.title("Sensitive Analysis of {} for {}".format(metric_name, param_name))
    plt.xlabel(sensitive_param)
    plt.plot(metric,label=metric_name)
    plt.legend()
    plt.savefig('output/sensitiveAnalysis/{}/{}.png'.format(param_name, metric_name))    

param = {}
param['pickup_distance_threshold'] = 2000
param['extra_distance_threshold'] = 3000

results = {}
results['extra_distance'] = []
results['saved_distance'] = []
results['taker_pickup_time'] = []
results['platflorm_income'] = []
results['shared_distance'] = []
results['dispatch_time'] = []
results['success_rate'] = []
results['waiting_time'] = []


pickup_distance_threshold = [i for i in range(1000, 6000, 500)]
extra_distance_threshold = [i for i in range(1000, 6000, 500)]
for i in range(len(pickup_distance_threshold)):
    param['extra_distance_threshold'] = extra_distance_threshold[i]
    # platform_income, or expected_shared_distance or passenger_experience
    res = Main.Main(optimazition_target = 'platform_income', matching_condition = True,\
    param = param ).run() 
    results['extra_distance'].append(res['extra_distance'])
    results['saved_distance'].append(res['saved_distance'])
    results['taker_pickup_time'].append(res['taker_pickup_time'])
    results['platflorm_income'].append(res['platflorm_income'])
    results['shared_distance'].append(res['shared_distance'])
    results['dispatch_time'].append(res['dispatch_time'])
    results['success_rate'].append(res['success_rate'])
    results['waiting_time'].append(res['waiting_time'])

for key in results.keys():
    plot_metrics(metric = results[key], sensitive_param = pickup_distance_threshold,\
         metric_name = key, param_name = 'extra_distance_threshold')

for i in range(len(pickup_distance_threshold)):
    param['pickup_distance_threshold'] = pickup_distance_threshold[i]
    # platform_income, or expected_shared_distance or passenger_experience
    res = Main.Main(optimazition_target = 'platform_income', matching_condition = True,\
    param = param ).run() 
    results['extra_distance'].append(res['extra_distance'])
    results['saved_distance'].append(res['saved_distance'])
    results['taker_pickup_time'].append(res['taker_pickup_time'])
    results['platflorm_income'].append(res['platflorm_income'])
    results['shared_distance'].append(res['shared_distance'])
    results['dispatch_time'].append(res['dispatch_time'])
    results['success_rate'].append(res['success_rate'])
    results['waiting_time'].append(res['waiting_time'])

for key in results.keys():
    plot_metrics(metric = results[key], sensitive_param = pickup_distance_threshold,\
         metric_name = key, param_name = 'pickup_distance_threshold')




# In[]

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

font = {'family' : 'Times New Roman', 'size'   : 16}

import pickle
path = "./output/2017-05-01/system_metric.pkl"

fr = open(path, 'rb')
dic = pickle.load(fr)
import pandas as pd
df = pd.DataFrame(list(zip(dic['ride_distance_error'],dic['shared_distance_error'])))
df.columns=['ride_distance_error','shared_distance_error']
df.head()
print(df['ride_distance_error'].describe())
print(df['shared_distance_error'].describe())
# In[]
plt.figure(figsize=[15,8])
plt.xlabel('ride_distance error')
plt.plot(dic['ride_distance_error'])
sns.distplot(dic['ride_distance_error'],kde = True, bins = 20,color="C0")
print(dic['ride_distance_error'])
plt.savefig('./ride_distance_error.png', dpi = 200)
plt.show()


# In[]
plt.xlabel('shared_distance error')
sns.distplot(dic['shared_distance_error'],kde = True, bins = 20,rug = True)
# In[]

import numpy as np
import Seeker


import pickle
path = "./input/expected_shared_distance/his_order.pkl"

fr = open(path, 'rb')
dic = pickle.load(fr)
# In[] 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

font = {'family' : 'Times New Roman', 'size'   : 16}

import pickle
path = "./output/expected_shared_distance/system_metric.pkl"

fr = open(path, 'rb')
dic = pickle.load(fr)

plt.grid(True)
kwargs = dict(histtype='stepfilled', alpha=0.5, normed=True, bins=40)

plt.figure(figsize=[15,8])
plt.xlabel('ride_distance error')
sns.distplot(dic['ride_distance_error'],kde = True, bins = 20,\
             kde_kws={"color": "C0", "lw": 2, "label": "KDE","alpha": 1},\
                 hist_kws={ "linewidth": 1,"alpha": 0.5, "color": "C0"})
sns.distplot(dic['shared_distance'],kde = True, bins = 20,\
             kde_kws={"color": "C1", "lw": 2, "label": "KDE","alpha": 1},\
                 hist_kws={ "linewidth": 1,"alpha": 0.5, "color": "C1"})
plt.ylabel('Frequency', fontdict=font)
plt.xlabel('Distance (m)', fontdict=font)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.legend(['ride distance for drivers','shared distance for passengers'],prop=font)
plt.savefig('./ride_distance_error.png', dpi = 200)
plt.show()
# In[] 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

font = {'family' : 'Times New Roman', 'size'   : 16}

import pickle
path = "./output/expected_shared_distance/system_metric.pkl"

fr = open(path, 'rb')
dic = pickle.load(fr)
print(dic['relative_ride_distance_error'])

data_to_plot = [dic['relative_ride_distance_error'], dic['relative_shared_distance_error']]


import matplotlib as mpl 
## agg backend is used to create plot as a .png file
mpl.use('agg')
# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# # Create the boxplot
# bp = ax.boxplot(data_to_plot)
# ## add patch_artist=True option to ax.boxplot() 
# ## to get fill color
# bp = ax.boxplot(data_to_plot, patch_artist=True)

bp = sns.violinplot( data=data_to_plot)
    
plt.ylabel('Relative Error', fontdict=font)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
# Save the figure
fig.savefig('./relative_ride_distance_error.png', dpi = 200)

plt.show()
# In[] 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

font = {'family' : 'Times New Roman', 'size'   : 16}

import pickle
path = "./output/2017-05-01/system_metric.pkl"

fr = open(path, 'rb')
dic = pickle.load(fr)
data_to_plot = [dic['ride_distance_error'], dic['shared_distance_error']]
import matplotlib as mpl 
## agg backend is used to create plot as a .png file
mpl.use('agg')
# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data_to_plot)
## add patch_artist=True option to ax.boxplot() 
## to get fill color
bp = ax.boxplot(data_to_plot, patch_artist=True)


    
plt.ylabel('Absolute Error', fontdict=font)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
# Save the figure
fig.savefig('./absolute_ride_distance_error.png', dpi = 200)

plt.show()
# In[]
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

font = {'family' : 'Times New Roman', 'size'   : 16}

import pickle
path = "./output/2017-05-01/system_metric.pkl"

fr = open(path, 'rb')
dic = pickle.load(fr)

# for i in range(len(dic['relative_ride_distance_error'])):
#     if dic['relative_ride_distance_error'][i] == 1:
#         dic['relative_ride_distance_error'][i] = \
#             np.mean(dic['relative_ride_distance_error'])
#         dic['ride_distance_error'][i] = \
#             np.mean(dic['ride_distance_error'])   
print(dic['relative_ride_distance_error'])
fr = open(path, 'rb')
dic = pickle.load(fr)
import pandas as pd
relative_error = pd.DataFrame(list(zip(dic['relative_ride_distance_error'],dic['relative_shared_distance_error'])))
relative_error.columns=['Ride Distance','Shared Distance']
relative_error.head()
print(relative_error['Ride Distance'].describe())
print(relative_error['Shared Distance'].describe())
import matplotlib as mpl 
## agg backend is used to create plot as a .png file
mpl.use('agg')
# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

sns.violinplot(
               data=relative_error,split=True,inner='quartiles')

    
plt.ylabel('Relative Error', fontdict=font)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.show()
# Save the figure
fig.savefig('./relative_error.png', dpi = 200)
plt.close()


# In[]
fr = open(path, 'rb')
dic = pickle.load(fr)
print(dic.keys())
import pandas as pd
absolute_error = pd.DataFrame(list(zip(dic['ride_distance_error'],dic['shared_distance_error'])))
absolute_error.columns=['Ride Distance','Shared Distance']
absolute_error.head()
print(absolute_error['Ride Distance'].describe())
print(absolute_error['Shared Distance'].describe())
import matplotlib as mpl 
## agg backend is used to create plot as a .png file
mpl.use('agg')
# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

sns.violinplot(
               data=absolute_error,split=True,inner='quartiles')

    
plt.ylabel('Absolute Error (m)', fontdict=font)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
# Save the figure
fig.savefig('./absolute_error.png', dpi = 200)

plt.show()
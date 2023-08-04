
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
    res = Main.Main(optimazition_target = 'expected_shared_distance', matching_condition = True,\
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








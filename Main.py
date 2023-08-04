
import pickle
import os
from common import Config
import importlib


class Main():

    def __init__(self, cfg) -> None:
        self.simu = sm.Simulation(cfg)

    def run(self):
        while True:
            reward, done = self.simu.step()
            if done:
                break
        res = self.simu.res
        value = res.get("waitingTime")
        print('waitingTime:', value)
        value = res.get("detour_distance")
        print('detour_distance:', value)
        value = res.get("pickup_time")
        print('pickup_time:', value)
        value = res.get("shared_distance")
        print('shared_distance:', value)
        value = res.get("total_ride_distance")
        print('total_ride_distance:', value)
        value = res.get("saved_ride_distance")
        print('saved_ride_distance:', value)
        value = res.get("platform_income")
        print('platform_income:', value)
        value = res.get("response_rate")
        print('response_rate:', value)
        value = res.get("carpool_rate")
        print('carpool_rate:', value)

        try:
            self.plot_metrics(metric=self.simu.detour_distance,
                              method=model, metric_name='detour_distance')
            self.plot_metrics(metric=self.simu.traveltime,
                              method=model, metric_name='traveltime')
            self.plot_metrics(metric=self.simu.waitingtime,
                              method=model, metric_name='waiting_time')
            self.plot_metrics(metric=self.simu.pickup_time,
                              method=model, metric_name='pickup_time')
            self.plot_metrics(metric=self.simu.platform_income,
                              method=model, metric_name='platform_income')
            self.plot_metrics(metric=self.simu.shared_distance,
                              method=model, metric_name='shared_distance')
        except:
            pass

        return res

    # 绘图
    def test_plot(self, rewards, smoothed_rewards, algo, env_name):

        self.plot_rewards(rewards, smoothed_rewards,  algo, env_name)

    # 绘图
    def plot_rewards(self, cfg, rewards, smoothed_rewards, algo='batchmatching', env_name='ridesharing'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        plt.figure()  # 创建一个图形实例，方便同时多画几个图
        plt.title("reward of {} for {}".format(algo, env_name))
        plt.xlabel('epsiodes')
        plt.plot(rewards, label='rewards')
        plt.plot(smoothed_rewards, label='smoothed rewards')
        plt.legend()
        # plt.savefig('output\\rewards.png')
        plt.savefig('output\\{}/rewards.png'.format(cfg.optimazition_target))

     # 绘图
    def plot_metrics(self, metric,  metric_name, method='batchmatching', env_name='ridesharing'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        plt.figure()  # 创建一个图形实例，方便同时多画几个图
        plt.title("{} of {} for {}".format(metric_name, method, env_name))
        plt.xlabel('orders')
        plt.plot(metric, label=metric_name)
        plt.legend()
        plt.savefig('output\\{}/{}.png'.format(method, metric_name))
        plt.close()


def different_ratio():
    ratios = [100/25,100/50,100/75]
    for ratio in ratios:
        cfg = Config.Config()
        cfg.order_driver_ratio = ratio
        cfg.progress_target = False
        print('ratio:',cfg.order_driver_ratio)
        import time
        start = time.time()
        ma = Main(cfg)
        ma.run() 
        end = time.time()
        print('执行时间{},order_driver_ratio:{}'.format(end - start, cfg.order_driver_ratio))
        print('ratio:',ma.simu.vehicle_num / len(ma.simu.order_list))

def different_date():
    import os
    files = os.listdir('input/orders/')
    dic = {}
    import pickle

    with open('output/results.pkl', "wb") as tf:
        for file in files:
            print(file)
            cfg = Config.Config()
            cfg.progress_target = False
            print('ratio:',cfg.order_driver_ratio)
            cfg.date = file.split('.')[0]
            cfg.order_file = 'input/orders/'+ file
            import time
            start = time.time()
            ma = Main(cfg).run() 
            end = time.time()
            print('file:{},执行时间:{}'.format(file, end - start))
            dic[file.split('.')[0]] = ma
        pickle.dump(dic, tf)

def one_day_test():
    cfg = Config.Config()
    print('ratio:',cfg.order_driver_ratio)
    import time
    start = time.time()
    ma = Main(cfg)
    ma.run() 
    end = time.time()
    print('执行时间{},order_driver_ratio:{}'.format(end - start, cfg.order_driver_ratio))
    print('ratio:',ma.simu.vehicle_num / len(ma.simu.order_list))


one_day_test()
# different_date()
# different_ratio()

model = "baseline2_batch_matching"
     # baseline1_first_protocol_consider_wait, baseline1_first_protocol, baseline2_batch_matching, baseline3_no_carpool,
     # baseline4_RTV, predicted_based_approach_saved_distance, predicted_based_approach
sm = importlib.import_module('method.'+model)

files = os.listdir('input/orders/')
print(files)
dic = {}
with open('output/{}/results.pkl'.format(model), "wb") as tf:
    for file in files:
        print(file)
        cfg = Config.Config()
        print('ratio:', cfg.order_driver_ratio)
        cfg.date = file.split('.')[0]
        cfg.order_file = 'input/orders/' + file
        import time
        start = time.time()
        ma = Main(cfg)
        ma.run()
        end = time.time()
        print('file:{},执行时间:{}'.format(file, end - start))
        dic[file.split('.')[0]] = ma

        folder = 'output/{}/{}/'.format(model, cfg.date)
        if not os.path.exists(folder):
            os.makedirs(folder)
        ma.simu.save_metric(path =folder + 'system_metric.pkl')
        ma.simu.save_his_order(path = folder + 'history_order.pkl')
    pickle.dump(dic, tf)

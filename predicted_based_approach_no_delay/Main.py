
import Simulation as sm
import Config


class Main():

    def __init__(self, cfg) -> None:
        self.simu = sm.Simulation(cfg)
        self.test_episodes = 1

    def run(self):
        rewards = []
        for i_ep in range(self.test_episodes):
            # print('episode:', i_ep)
            ep_reward = 0  # 一个回合的奖励
            while True:
                reward, done = self.simu.step()
                # print(self.simu.time)
                # print('reward', reward)
                ep_reward += reward
                if done:
                    break
            rewards.append(ep_reward)

            # print('rewards',rewards,smooth_rewards,'smooth_rewards')
        res = self.simu.res
        print('waitingTime:', res['waitingTime'])
        print('detour_distance:', res['detour_distance'])
        print('pickup_time:', res['pickup_time'])
        print('shared_distance:', res['shared_distance'])
        print('total_ride_distance:', res['total_ride_distance'])
        print('saved_ride_distance:', res['saved_ride_distance'])
        print('platform_income:', res['platform_income'])
        print('response_rate:', res['response_rate'])
        print('carpool_rate:', res['carpool_rate'])

        self.plot_metrics(
            self.simu.cfg, metric=self.simu.detour_distance, metric_name='detour_distance')
        self.plot_metrics(
            self.simu.cfg, metric=self.simu.traveltime, metric_name='traveltime')
        self.plot_metrics(
            self.simu.cfg, metric=self.simu.waitingtime, metric_name='waiting_time')
        self.plot_metrics(
            self.simu.cfg, metric=self.simu.pickup_time, metric_name='pickup_time')
        self.plot_metrics(
            self.simu.cfg, metric=self.simu.platform_income, metric_name='platform_income')
        self.plot_metrics(
            self.simu.cfg, metric=self.simu.shared_distance, metric_name='shared_distance')

        return res

     # 绘图
    def plot_metrics(self, cfg, metric,  metric_name, algo='batchmatching', env_name='ridesharing'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        plt.figure()  # 创建一个图形实例，方便同时多画几个图
        plt.title("{} of {} for {}".format(metric_name, algo, env_name))
        plt.xlabel('orders')
        plt.plot(metric, label=metric_name)
        plt.legend()
        plt.savefig(
            'output\\{}/{}.png'.format(cfg.optimazition_target, metric_name))
        plt.close()


def different_ratio():
    ratios = [100/10, 100/50]
    for ratio in ratios:
        cfg = Config.Config()
        cfg.order_driver_ratio = ratio
        cfg.progress_target = False
        print('ratio:', cfg.order_driver_ratio)
        import time
        start = time.time()
        ma = Main(cfg)
        ma.run()
        end = time.time()
        print('执行时间{},order_driver_ratio:{}'.format(
            end - start, cfg.order_driver_ratio))


def different_ratio_profit():
    ratios = [100/10, 100/25]
    for ratio in ratios:
        cfg = Config.Config()
        cfg.order_driver_ratio = ratio
        # platform_income, expected_shared_distance,combination
        cfg.optimazition_target = 'platform_income'
        cfg.progress_target = False
        print('ratio:', cfg.order_driver_ratio,
              'optimazition_target', cfg.optimazition_target)
        import time
        start = time.time()
        ma = Main(cfg)
        ma.run()
        end = time.time()
        print('执行时间{},order_driver_ratio:{}'.format(
            end - start, cfg.order_driver_ratio))
        print('ratio:', ma.simu.vehicle_num / len(ma.simu.order_list))


def different_date():
    import os
    files = os.listdir('input/orders/')
    files = ['2017-05-04.csv', '2017-05-05.csv', '2017-05-06.csv', '2017-05-07.csv', '2017-05-08.csv', '2017-05-09.csv', '2017-05-10.csv', '2017-05-11.csv', '2017-05-12.csv',
             '2017-05-13.csv', '2017-05-14.csv', '2017-05-15.csv', '2017-05-16.csv', '2017-05-17.csv', '2017-05-18.csv', '2017-05-19.csv', '2017-05-20.csv', '2017-05-21.csv']
    dic = {}
    import pickle

    with open('output/results.pkl', "wb") as tf:
        for file in files:
            print(file)
            cfg = Config.Config()
            cfg.progress_target = False
            print('ratio:', cfg.order_driver_ratio)
            cfg.date = file.split('.')[0]
            cfg.order_file = 'input/orders/' + file
            import time
            start = time.time()
            ma = Main(cfg).run()
            end = time.time()
            print('file:{},执行时间:{}'.format(file, end - start))
            dic[file.split('.')[0]] = ma
        pickle.dump(dic, tf)


def different_hour():
    import os
    files = os.listdir('input/hours/matched_orders/')
    for i in range(len(files)):
        cfg = Config.Config()
        cfg.progress_target = False

        cfg.order_file = 'input/hours/matched_orders/' + files[i]
        cfg.simulation_begin_time = ' ' + files[i][6:8] + ':00:00'
        if i == len(files) - 1:
            cfg.simulation_end_time = ' 24:00:00'
        else:
            cfg.simulation_end_time = ' ' + files[i+1][6:8] + ':00:00'
        print(cfg.simulation_begin_time,cfg.simulation_end_time)
        import time
        start = time.time()
        ma = Main(cfg).run()
        end = time.time()
        print('file:{},执行时间:{}'.format(files[i], end - start))


def different_rw():
    rws = [0.55, 0.74]
    for rw in rws:
        cfg = Config.Config()
        cfg.rw = rw
        cfg.progress_target = False
        print('rw:', cfg.rw)
        import time
        start = time.time()
        ma = Main(cfg)
        ma.run()
        end = time.time()
        print('执行时间{},rw:{}'.format(end - start, cfg.rw))


def one_day_test():
    cfg = Config.Config()
    # platform_income, expected_shared_distance,combination
    cfg.optimazition_target = 'expected_shared_distance'
    print('ratio:', cfg.order_driver_ratio,
          'optimazition_target:', cfg.optimazition_target, )
    import time
    start = time.time()
    ma = Main(cfg)
    ma.run()
    end = time.time()
    print('执行时间{},order_driver_ratio:{}'.format(
        end - start, cfg.order_driver_ratio))


def multi_obj_test():
    cfg = Config.Config()
    cfg.optimazition_target = 'combination'
    print('optimazition_target:', cfg.optimazition_target, )
    import time
    start = time.time()
    ma = Main(cfg)
    ma.run()
    end = time.time()
    print('执行时间{},order_driver_ratio:{}'.format(
        end - start, cfg.order_driver_ratio))
    print('beta', cfg.beta)


def profit_test():
    cfg = Config.Config()
    # platform_income, expected_shared_distance,combination
    cfg.optimazition_target = 'platform_income'
    # cfg.order_driver_ratio = 100/10
    print('optimazition_target:', cfg.optimazition_target, )
    import time
    start = time.time()
    ma = Main(cfg)
    ma.run()
    end = time.time()
    print('执行时间{},order_driver_ratio:{}'.format(
        end - start, cfg.order_driver_ratio))


def different_beta():
    import numpy as np
    betas = np.arange(0.1, 1, 0.1)
    for beta in betas:
        cfg = Config.Config()
        cfg.beta = beta
        cfg.progress_target = False
        cfg.optimazition_target = 'combination'
        print('optimazition_target:', cfg.optimazition_target, 'beta:', cfg.beta)
        cfg.delay_time_threshold = 60
        import time
        start = time.time()
        ma = Main(cfg)
        ma.run()
        end = time.time()
        print('执行时间{},beta:{}'.format(end - start, cfg.beta))


def different_K():
    Ks = [400, 500, 600]
    for delay_time_threshold in Ks:
        cfg = Config.Config()
        cfg.delay_time_threshold = delay_time_threshold
        cfg.progress_target = False
        print('delay_time_threshold:', cfg.delay_time_threshold)
        import time
        start = time.time()
        ma = Main(cfg)
        ma.run()
        end = time.time()
        print('执行时间{},delay_time_threshold:{}'.format(
            end - start, cfg.delay_time_threshold))


def different_time_window():
    time_windows = [2, 30, 60]
    for time_window in time_windows:
        cfg = Config.Config()
        cfg.time_unit = time_window
        cfg.progress_target = False
        print('time_unit:', cfg.time_unit)
        import time
        start = time.time()
        ma = Main(cfg)
        ma.run()
        end = time.time()
        print('执行时间{},time_unit:{}'.format(end - start, cfg.time_unit))

one_day_test()
# different_date()
# different_ratio()
# different_rw()
# multi_obj_test()
# profit_test()
# different_beta()
# different_K()
# different_ratio_profit()
# different_time_window()


# different_hour()

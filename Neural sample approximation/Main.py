
from statistics import mean
from regex import FULLCASE
import Simulation as sm




class Main():

    def __init__(self,optimazition_target = 'platform_income', matching_condition = True, param = []) -> None:
        self.optimazition_target = optimazition_target
        self.simu = sm.Simulation(optimazition_target = optimazition_target, matching_condition = matching_condition,\
             param = [])
        self.test_episodes = 1
 
 
    def run(self):
        rewards = []
        for i_ep in range(self.test_episodes):
            print('episode:', i_ep)
            ep_reward = 0 # 一个回合的奖励
            self.simu.reset()
            while True:
                reward, done = self.simu.step()
                # print(self.simu.time)
                # print('reward', reward)
                ep_reward += reward
                if done:
                    break
            rewards.append(ep_reward)

            # print('rewards',rewards,smooth_rewards,'smooth_rewards')
            self.simu.save_metric(path = "output/{}/system_metric.pkl".format(self.optimazition_target))
            print('success_rate', self.simu.success_rate) 
        # self.test_plot(rewards , smooth_rewards, algo = 'first_protocol matching', env_name='ridesharing')
        # self.plot_metrics(metric = self.simu.extra_distance, metric_name = 'extra_distance')
        # self.plot_metrics(metric = self.simu.saved_distance, metric_name = 'saved_distance')
        # self.plot_metrics(metric = self.simu.waiting_time, metric_name = 'waiting_time')
        # self.plot_metrics(metric = self.simu.taker_pickup_time, metric_name = 'taker_pickup_time')
        # self.plot_metrics(metric = self.simu.platflorm_income, metric_name = 'platflorm_income')
        # self.plot_metrics(metric = self.simu.shared_distance, metric_name = 'shared_distance')
        # self.plot_metrics(metric = self.simu.dispatch_time, metric_name = 'dispatch_time')
        res = {}
        res['extra_distance'] = mean(self.simu.extra_distance)
        res['saved_distance'] = mean(self.simu.saved_distance)
        res['taker_pickup_time'] = mean(self.simu.taker_pickup_time)
        res['waiting_time'] = mean(self.simu.waiting_time)
        res['platflorm_income'] = mean(self.simu.platflorm_income)
        res['shared_distance'] = mean(self.simu.shared_distance)
        res['dispatch_time'] = mean(self.simu.dispatch_time)
        res['success_rate'] = self.simu.success_rate
        
        return res

    # 绘图
    def test_plot(self,rewards, smoothed_rewards, algo, env_name):
 
        self.plot_rewards(rewards, smoothed_rewards,  algo, env_name)
 
 
    # 绘图
    def plot_rewards(self, rewards,smoothed_rewards, algo = 'batchmatching', env_name='ridesharing'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        plt.figure() # 创建一个图形实例，方便同时多画几个图
        plt.title("reward of {} for {}".format(algo,env_name))
        plt.xlabel('epsiodes')
        plt.plot(rewards,label='rewards')
        plt.plot(smoothed_rewards,label='smoothed rewards')
        plt.legend()
        # plt.savefig('output\\rewards.png')
        plt.savefig('output\\{}/rewards.png'.format(self.optimazition_target))      

     # 绘图
    def plot_metrics(self,metric,  metric_name, algo = 'batchmatching', env_name='ridesharing'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        plt.figure() # 创建一个图形实例，方便同时多画几个图
        plt.title("{} of {} for {}".format(metric_name, algo,env_name))
        plt.xlabel('orders')
        plt.plot(metric,label=metric_name)
        plt.legend()
        plt.savefig('output\\{}/{}.png'.format(self.optimazition_target, metric_name))       

# platform_income, or expected_shared_distance or passenger_experience
Main(optimazition_target = 'expected_shared_distance', matching_condition = True).run() 

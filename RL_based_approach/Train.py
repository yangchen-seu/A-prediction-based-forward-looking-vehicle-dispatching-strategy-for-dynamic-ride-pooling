
import sys, os
import numpy as np

# curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前路径
# parent_path=os.path.dirname(curr_path) # 父路径，这里就是我们的项目路径
# sys.path.append(parent_path) # 由于需要引用项目路径下的其他模块比如envs，所以需要添加路径到sys.path

import datetime
import Config
import Ridesharing_env
import Vehicle
import random
from common.model import Critic
from common.memory import ReplayBuffer

cfg = Config.Config()

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

def __init__(self,):
    pass


def env_agent_config(cfg, vehicle_num = 100, seed = 1):


    vehicle_lis = []
    # 初始化critic
    critic = Critic(cfg.n_states, cfg)
    # 初始化多智能体
    for driver_id in range(vehicle_num):
        agent = Vehicle.Vehicle(driver_id, random.choice(cfg.all_locations),cfg)
        vehicle_lis.append(agent)

    # 初始化环境
    env = Ridesharing_env.Ridesharing_env(vehicle_lis)
    print('环境初始化完毕')
    return env, vehicle_lis, critic

def train(cfg, env, vehicle_lis, critic):
    memory = ReplayBuffer(cfg.memory_capacity) # 经验回放
    # 训练
    print('begin training')
    print(f'env:{cfg.env_name}, algo:{cfg.algo}, device:{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    smooth_rewards = [] # 平滑奖励
    for i_ep in range(cfg.train_eps):
        print('episode:', i_ep)
        ep_reward = 0 # 一个回合的奖励
        time_slot = env.reset(vehicle_lis) # 重置环境，返回初始状态
        while True:
            memory_agents = []
            for agent in vehicle_lis:
                if agent.state == 1:
                    # print('agent{} action'.format(agent.id))    
                    agent.value = critic.value_net([agent.location, time_slot])
                    agent.target_value = critic.target_net([agent.location, time_slot])
                    memory_agents.append(agent)



            reward, done = env.step(vehicle_lis) # 更新环境，返回transition

            # 存储经验
            for agent in memory_agents:
                memory.push(agent.origin_location,  agent.reward, \
                    agent.location, done) # 存入memory中


            critic.update(memory)  # 智能体更新

            ep_reward += reward # 累加奖励

            if done:
                break

        # env.render()
        if (i_ep + 1)% cfg.target_update == 0: # 更新目标网络
            critic.target_net.load_state_dict(critic.value_net.state_dict())
        
        # if (i_ep + 1) % 10 == 0:
        print('回合:{}/{}, 奖励：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        if smooth_rewards:
            smooth_rewards.append(0.9 * smooth_rewards[-1] + 0.1 * ep_reward)
        else:
            smooth_rewards.append(ep_reward)

    print('finish training')
    return rewards, smooth_rewards

def test(cfg, env, agent):
    print('begin testing')
    print(f'env:{cfg.env}, algo:{cfg.algo}, device:{cfg.device}')
    # 测试时不需要使用epsilon-greedy策略，所以相应的值设为0
    cfg.epsilon_start = 0.0 
    cfg.epsilon_end = 0.0
    rewards = []
    smooth_rewards = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0 # 记录一回合的奖励
        state = env.reset() # 重置环境
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if smooth_rewards:
            smooth_rewards.append(smooth_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            smooth_rewards.append(ep_reward)
        print(f'episode:{i_ep + 1}/{cfg.test_eps}, reward:{ep_reward:.1f}')
    print('test finished')
    return rewards, smooth_rewards



def train_plot(cfg, agent, rewards, smoothed_rewards):
    os.makedirs(cfg.result_path,exist_ok=True)
    os.makedirs(cfg.model_path,exist_ok=True)
    agent.save(path = cfg.model_path) 
    save_results(rewards, smoothed_rewards, tag='train', path = cfg.result_path)
    plot_rewards(rewards, smoothed_rewards, device = cfg.device, \
        algo = cfg.algo, env_name = cfg.env, tag = 'train'
        )

def test_plot(cfg, agent, rewards, smoothed_rewards):
    os.makedirs(cfg.result_path,exist_ok=True)
    os.makedirs(cfg.model_path,exist_ok=True)
    agent.save(path = cfg.model_path) 
    save_results(rewards, smoothed_rewards, tag='test', path = cfg.result_path)
    plot_rewards(rewards, smoothed_rewards, device = cfg.device, \
        algo = cfg.algo, env_name = cfg.env, tag = 'test'
        )

def plot_rewards(rewards,smoothed_rewards,device, algo,env_name,tag='train'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set() 
    plt.figure() # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(device, algo,env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards,label='rewards')
    plt.plot(smoothed_rewards,label='smoothed rewards')
    plt.legend()
    plt.show()

def save_results(rewards,smooth_rewards,tag='train',path = '\\outputs\\results\\'):
    '''save rewards and smooth_rewards
    '''
    np.save(path +'{}_rewards.npy'.format(tag), rewards)
    np.save( path+'{}_smooth_rewards.npy'.format(tag), smooth_rewards)
    print('结果保存完毕！')


if __name__ == '__main__':
    cfg = Config.Config()
    env, vehicle_lis, critic = env_agent_config(cfg, seed = 1)
    # training
    rewards, smooth_rewards = train(cfg, env, vehicle_lis,critic)
    plot_rewards(rewards,smooth_rewards,cfg.device, cfg.algo,cfg.env_name,tag='train')

    # testing
    # agent.load(path = cfg.model_path)
    # rewards, smooth_rewards = test(cfg, env, agent)
    # test_plot(cfg, agent, rewards, smooth_rewards)
    # env = gym.make('Ridesharing_env-v0')
    # env.step()
    # env.reset()
        # save object
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

    path = os.path.abspath(os.path.dirname(__file__)) + '\\output\\critic.pickle'
    def save(path):
        import pickle
        with open(path, 'wb') as handle:
            pickle.dump(critic, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
       
#     save(path)

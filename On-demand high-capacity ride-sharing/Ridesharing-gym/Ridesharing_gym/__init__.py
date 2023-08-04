'''
Author: your name
Date: 2021-12-07 21:41:20
LastEditTime: 2021-12-07 21:42:23
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \强化学习网约车\Reinforcementlearning\env\__init__.py
'''
from gym.envs.registration import register

register(
	id = 'Ridesharing_env-v0', # 环境名,版本号v0必须有
	entry_point = 'Ridesharing_gym.envs:Ridesharing_env' , # 文件夹名.文件名:类名
	# 根据需要定义其他参数
)
# A-prediction-based-forward-looking-vehicle-dispatching-strategy-for-dynamic-ride-pooling
The code for paper titled 'A prediction-based forward-looking vehicle dispatching strategy for dynamic ride-pooling'
## Readme
### 模块说明
- Neural sample approximation， 一个基于神经网络的价值函数近似的匹配权重计算方法，参考论文：Neural Approximate Dynamic Programming for On-Demand Ride-Pooling
- On-demand high-capacity ride-sharing ，论文的一个baseline，参考：On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment
- RL_based_approach，未完成的强化学习派单方法
- analysis，做的一些数据分析
- baseline1_first_protocol，论文的baseline，立即匹配算法
- baseline2_batch_matching，论文的baseline，短视的批匹配算法
- baseline3_no_carpool，论文的baseline，不考虑拼车下的匹配算法
- common，一些基础组件，主要用在RL上
- input，通用的输入信息
- known_model，论文的baseline，假设所有供需已知的全知模型
- maximum matching，求解最大匹配问题
- method，一些基础组件，主要用在RL上
- output，通用的输出信息
- predicted_based_approach_no_delay，论文的baseline，不考虑delayed matching的匹配算法
- predicted_based_approach_saved_distance，论文提出的算法，主要看这个文件夹
- robust model，拓展到鲁棒优化的派单算法

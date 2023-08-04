'''
Author: your name
Date: 2022-03-09 15:49:01
LastEditTime: 2022-03-09 16:15:15
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \强化学习网约车\mean-field\common\Km.py
'''
import numpy as np


class Km():
	def __init__(self, Matrix):
		Num = len(Matrix)
		self.E_val = Matrix # 记录每条边的权值
		self.L_val = [0 for _ in range(Num)] # 每个左顶点的值
		self.R_val = [0 for _ in range(Num)] # 每个右顶点的值
		self.L_vis = [False for _ in range(Num)] # 记录每一轮匹配匹配过的左顶点
		self.R_vis = [False for _ in range(Num)] # 记录每一轮匹配匹配过的右顶点
		self.match = [-1 for _ in range(Num)] # 记录每个右顶点匹配到的左顶点 如果没有则为-1
		self.slack = [0 for _ in range(Num)] # 记录每个右顶点如果能被左顶点匹配最少还需要多少值
		self.N = Num # 单边顶点数量

	def dfs(self, now):
		self.L_vis[now] = True
		for i in range(self.N):
			if self.R_vis[i]: # 每一轮匹配 每个右顶点只尝试一次
				continue
			tmp = self.L_val[now] + self.R_val[now] - self.E_val[now][i]

			if tmp == 0: #如果符合要求
				self.R_vis[i] = True
				if self.match[i] == -1 or self.dfs(self.match[i]): # 找到一个没有匹配的右顶点 或者该右顶点当前匹配的左顶点可以找到其它匹配
					self.match[i] = now
					return True
			else:
				self.slack[i] = min(self.slack[i] , tmp)
		return False


	def run(self):
		N = self.N
		self.match = [-1 for _ in range(self.N)]
		self.R_val = [0 for _ in range(self.N)]
		for i in range(N):
			self.L_val[i] = self.E_val[i][0]
			for j in range(1,N):
				self.L_val[i] = max(self.L_val[i] , self.E_val[i][j])


		# 尝试为每一个左顶点匹配
		for i in range(N):
			print('i',i)
			self.slack = [float('inf') for _ in range(N)] # 取最小值，就初始化为无穷大
			while True:
				# 为每个左顶点匹配的方法是 ：如果找不到就降低期望值，直到找到为止
            	# 记录每轮匹配中左右顶点是否被尝试匹配过
				self.L_vis = [False for _ in range(N)] 
				self.R_vis = [False for _ in range(N)]
				if self.dfs(i): break # 找到匹配 退出
            	# 如果不能找到 就降低期望值
            	# 最小可降低的期望值
				d = float('inf')
				for j in range(N):
					if not self.R_vis[j]: d = min(d, self.slack[j])
				for j in range(N):
            		# 所有访问过的(被涉及的)左顶点降低值
					if self.L_vis[j]: 
						self.L_val[j] -= d
	                # 所有访问过(被涉及的)的右顶点增加值
					if self.R_vis[j]: 
						self.R_val[j] += d
	                # 没有访问过的右顶点 因为左顶点的期望值降低，距离被左顶点匹配又进了一步
					else: self.slack[j] -= d

	    # 匹配完成 求出所有匹配的权值和
		print('i',i)
		res = 0
		for i in range(N):
			res += self.E_val[ self.match[i] ][i]
		return res

# matrix = [
# 	[62, 41, 86, 94],
# 	[73, 58, 11, 12],
# 	[69, 93, 89, 88],
# 	[81, 40, 69, 13]
# ]
# km = KM(4, matrix)
# res = km.run()
# print(res)
# print(np.sum(matrix)  - res)


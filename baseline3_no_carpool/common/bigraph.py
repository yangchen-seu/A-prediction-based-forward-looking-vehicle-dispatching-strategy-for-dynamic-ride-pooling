'''
Author: your name
Date: 2022-03-24 21:22:28
LastEditTime: 2022-03-24 21:22:28
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \强化学习网约车\IDQN\common\bigraph.py
'''
'''
Author: your name
Date: 2022-03-09 15:48:55
LastEditTime: 2022-03-24 21:14:40
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \强化学习网约车\common\Km.py
'''

import numpy as np
import sys

class bigraph():
	def __init__(self, Matrix):
		self.Matrix = np.mat(Matrix) # 记录每条边的权值
		self.match = [-1 for _ in range(len(self.Matrix))] # 记录每个右顶点匹配到的左顶点 如果没有则为-1
		self.res = []
		self.N = len(self.Matrix)	 # 单边顶点数量
		self.M = 1e6

	def step1(self):
		# 每一行减去本行元素最小值
		matrix = self.Matrix - self.Matrix.min(axis = 1)
		# print(matrix)
		# 每一列减去本行元素最小值
		matrix = matrix - matrix.min(axis = 0)
		print(matrix)
		
		return matrix

	def step2(self):
		while True:
			matrix = self.Matrix.copy()
			# 确定独立零元素
			count = 0
			while True :
				target_matrix = matrix.copy()
				count += 1
				for i in range(len(matrix)):
					if np.count_nonzero(matrix[i]) == self.N - 1 :# 0元素只有一个
						zero_index = matrix[i].argmin()

						matrix[:,zero_index] = self.M

						matrix[i,zero_index] = 0
						# print('matrix hang',matrix)
			
				for j in range(len(matrix)):
					col = matrix[:,j]
					
					if np.count_nonzero(col) == self.N - 1 :# 0元素只有一个
						zero_index = col.argmin()
						# print('zero_index',zero_index)
						matrix[zero_index,:] = self.M
						# print('matrix lie',matrix)
						matrix[zero_index, j] = 0
						# print('matrix lie',matrix)
				# 如果有多个零元素
				for i in range(len(matrix)):
					if np.count_nonzero(matrix[i]) < self.N  :# 有0元素
						zero_index = matrix[i].argmin()

						matrix[:,zero_index] = self.M

						matrix[i,zero_index] = 0
						# print('matrix hang',matrix)
			
				for j in range(len(matrix)):
					col = matrix[:,j]
					
					if np.count_nonzero(col) < self.N  :# 有0元素
						zero_index = col.argmin()
						# print('zero_index',zero_index)
						matrix[zero_index,:] = self.M
						# print('matrix lie',matrix)
						matrix[zero_index, j] = 0
						# print('matrix lie',matrix)

				if  (matrix == target_matrix).all():
					print('迭代停止,count=',count)
					break
				
			# print('matrix',matrix) 
			# sys.exit()
			
			# 判断是否达到最优解
			if np.count_nonzero(matrix) == len(matrix) * (len(matrix) - 1):
				break

			else:
				res = np.where(matrix == 0)
				
				# 记录没有独立零元素的行
				row = [i for i in range(self.N) if i not in np.where(matrix == 0)[0]]
				col = []
				
				col_target = 0
				row_target = 0
				while True:
					for j in range(len(row)):
						# index = np.argmin(self.Matrix[row[j], :])
						index = np.where (self.Matrix[row[j], :] == 0)[1]
						for tmp in index:
							if tmp not in col:
								col.append(tmp)  # 被删除的0元素所在的列

					for j in range(len(col)):
						for i in range(len(res[0])):
							if res[1][i] == col[j]: # 当前打钩的列
								if res[0][i] not in row:
									row.append(res[0][i])
					# print('self.Matrix',self.Matrix)
					

					if col_target == len(col) and row_target == len(row) : # 没有新的标记
						print('row,col',row,col)
						print('len(row),len(col)',len(row),len(col))

						# 找到可以操作的剩余数据，包括打钩的行和没有钩的列
						# 没有钩的列和行
						col_ = [i for i in range(self.N) if i not in col]
						row_ = [i for i in range(self.N) if i not in row]

						# 剩余数据
						remaind = self.Matrix[ row, :]
						remaind = np.array(remaind[:, col_])

						# if np.any(remaind == 0):
						# 	print(remaind)
							
							
						# 	#简单方法
						# 	num = sum(remaind.flatten() == 0)
						# 	print('0元素个数：',  num)
						# 	sys.exit()
						# print(remaind)
						# sys.exit()
						if len(remaind) <= len(remaind[0]):
							# print(col,"col",row,"row")
							# print('self.Matrix',self.Matrix)
							self.Matrix[row] = self.Matrix[row] - remaind.min()
							# print('self.Matrix',self.Matrix)
							self.Matrix[:,col] = self.Matrix[:,col] + remaind.min()
							# print('self.Matrix',self.Matrix)
							# sys.exit()
						else:
							self.Matrix[:, col_] = self.Matrix[:, col_] - remaind.min()
							self.Matrix[row_] = self.Matrix[row_] + remaind.min()

						break
					else :
						col_target = len(col) 
						row_target = len(row)
		
								
		return np.where(matrix == 0)

	def run(self):
		self.Matrix = self.step1()
		res = self.step2()

		# print('res', res)
		return res



		
# matrix = [
# 	[62, 41, 86, 94],
# 	[73, 58, 11, 12],
# 	[69, 93, 89, 88],
# 	[81, 40, 69, 13]
# ]

# matrix = [
# 	[62, 41, 1, -5],
# 	[73, 58, -5,-5],
# 	[69, 93, -5, -5],
# 	[81, 40, -5, -5]
# ]
# matrix  = np.random.rand(100,100)
# matrix = [
# 	[4, 8, 7, 0, 0],
# 	[7, 9, 17, 0, 0],
# 	[6, 9, 12, 0, 0],
# 	[6, 7, 14,0, 0],
# 	[0, 0, 0, 0, 0]
# ]
# matrix = [
# 	[4, 8, 7, 15, 12,6],
# 	[7, 9, 17, 14, 10,30],
# 	[6, 9, 12, 8, 7,5],
# 	[6, 7, 14, 6, 10,6],
# 	[1, 2, 5, 7, 10,6],
# 	[6, 9, 12, 10, 6,8]
# ]

# matrix = np.mat(matrix)
# print(np.count_nonzero(matrix[:,0]))

# 测试拼车数据
import os
file = os.getcwd() + '\\IDQN\\output\\matrix.npy'
matrix = np.load(file)



# print(matrix.min(axis = 1))
# print(matrix.min(axis = 0))
# print(matrix - matrix.min(axis = 0))


import time
begin = time.time()
km = bigraph( matrix)
km.run()
end = time.time()
print('计算时间', end - begin)



# matrix = np.mat(matrix)
# print( matrix- 5)
# print(matrix)


# tmp1 = np.mat(km.Matrix)
# print('初始化',tmp1)
# print('行减',tmp1 - tmp1.min(axis = 1))
# print('列减',km.step1())


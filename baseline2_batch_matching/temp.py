import conda
import matplotlib.pyplot as plt
import numpy as np
# In[]
lambda_ = 1.1
gamma = 600
x = np.linspace(0,30)

y1 = 1000 * 1.1 ** x
y2 = 3000 - gamma * x

# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.show()


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
 
fig = plt.figure()
ax3 = plt.axes(projection='3d')

plt.rcParams['font.sans-serif']=['FangSong'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

#定义三维数据
xx = np.arange(1,30,1)
yy = np.arange(10,600,10)
X, Y = np.meshgrid(xx, yy)

Z = 1800 -  Y * X - 1000 * 1.05 ** X

#作图
ax3.plot_surface(X,Y,Z,cmap='rainbow') 
# ax3.contour(X,Y,Z,zdir="x",offset=1,cmap="rainbow")   #x轴投影
# ax3.contour(X,Y,Z,zdir="y",offset=10,cmap="rainbow")    #y轴投影
ax3.contour(X,Y,Z,[0],zdir="z",offset=-2500,cmap="rainbow")   #z轴投影

# 改变cmap参数可以控制三维曲面的颜色组合, 一般我们见到的三维曲面就是 rainbow 的

plt.show()
# In[]
import pandas as pd
order = pd.read_csv('./input/order.csv')
order.describe()
# %%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pickle
path = "./output/expected_shared_distance/system_metric.pkl"

fr = open(path, 'rb')
dic = pickle.load(fr)

plt.xlabel('ride_distance')
sns.distplot(dic['ride_distance_error'],kde = True, bins = 20,rug = True)
plt.show()
# In[]
plt.xlabel('shared_distance')
sns.distplot(dic['shared_distance_error'],kde = True, bins = 20,rug = True)
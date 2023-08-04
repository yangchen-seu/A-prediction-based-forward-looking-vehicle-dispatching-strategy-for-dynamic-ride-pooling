'''
Author: yangchen-seu 58383528+yangchen-seu@users.noreply.github.com
Date: 2023-01-04 15:27:47
LastEditors: yangchen-seu 58383528+yangchen-seu@users.noreply.github.com
LastEditTime: 2023-01-12 02:06:21
FilePath: \forward-looking-ridepooling-matching-algorithm\code\matching\predicted_based_approach_saved_distance\parameter.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
 
 
fig = plt.figure()
ax = fig.gca(projection='3d')
 
# Make data.
X = np.arange(1, 30, 1)
Y = np.arange(0, 600, 50)



X, Y = np.meshgrid(X, Y)


lambda_ = 1.01
Es0_ = 950
Es1_ = 1780



R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
 
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
 
# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
 
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
 
plt.show()
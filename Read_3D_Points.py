from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys

fig = pyplot.figure()
ax = Axes3D(fig)

# input_file = open('/media/user/ssd_1TB/YCB_dataset/points/002_master_chef_can/points.xyz')
# input_file = open('/media/user/ssd_1TB/YCB_dataset/points/025_mug/points.xyz')
input_file = open('/media/user/ssd_1TB/YCB_dataset/models/025_mug/textured.obj')
cld = []
x = []
y = []
z = []
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    input_line = input_line[:-1]
    input_line = input_line.split(' ')
    # print(input_line)
    if input_line[0] == 'v':
        cld.append([float(input_line[1]), float(input_line[2]), float(input_line[3])])
        x.append(float(input_line[1]))
        y.append(float(input_line[2]))
        z.append(float(input_line[3]))
    if input_line[0] == 'vt' or input_line[0] == 'vn':
        break
input_file.close()

x_max = np.max(x)
x_min = np.min(x)
y_max = np.max(y)
y_min = np.min(y)
z_max = np.max(z)
z_min = np.min(z)

corners = np.array([[x_min, y_min, z_min],
                    [x_max, y_min, z_min],
                    [x_max, y_max, z_min],
                    [x_min, y_max, z_min],

                    [x_min, y_min, z_max],
                    [x_max, y_min, z_max],
                    [x_max, y_max, z_max],
                    [x_min, y_max, z_max]])

print(corners)

ax.plot([corners[0, 0], corners[1, 0]], [corners[0, 1], corners[1, 1]],
        zs=[corners[0, 2], corners[1, 2]], c='r')
ax.plot([corners[1, 0], corners[2, 0]], [corners[1, 1], corners[2, 1]],
        zs=[corners[1, 2], corners[2, 2]], c='r')
ax.plot([corners[2, 0], corners[3, 0]], [corners[2, 1], corners[3, 1]],
        zs=[corners[2, 2], corners[3, 2]], c='r')
ax.plot([corners[3, 0], corners[0, 0]], [corners[3, 1], corners[0, 1]],
        zs=[corners[3, 2], corners[0, 2]], c='r')

ax.plot([corners[4, 0], corners[5, 0]], [corners[4, 1], corners[5, 1]],
        zs=[corners[4, 2], corners[5, 2]], c='r')
ax.plot([corners[5, 0], corners[6, 0]], [corners[5, 1], corners[6, 1]],
        zs=[corners[5, 2], corners[6, 2]], c='r')
ax.plot([corners[6, 0], corners[7, 0]], [corners[6, 1], corners[7, 1]],
        zs=[corners[6, 2], corners[7, 2]], c='r')
ax.plot([corners[7, 0], corners[4, 0]], [corners[7, 1], corners[4, 1]],
        zs=[corners[7, 2], corners[4, 2]], c='r')

ax.plot([corners[0, 0], corners[4, 0]], [corners[0, 1], corners[4, 1]],
        zs=[corners[0, 2], corners[4, 2]], c='r')
ax.plot([corners[1, 0], corners[5, 0]], [corners[1, 1], corners[5, 1]],
        zs=[corners[1, 2], corners[5, 2]], c='r')
ax.plot([corners[2, 0], corners[6, 0]], [corners[2, 1], corners[6, 1]],
        zs=[corners[2, 2], corners[6, 2]], c='r')
ax.plot([corners[3, 0], corners[7, 0]], [corners[3, 1], corners[7, 1]],
        zs=[corners[3, 2], corners[7, 2]], c='r')

ax.scatter(x, y, z, s=1, depthshade=False)
pyplot.show()
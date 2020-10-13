from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys
from PIL import Image

cx = 319.077
cy = 242.885
fx = 388.158
fy = 388.158
cam_scale = 0.0010000000474974513

K = [[fx, 0 , cx],
     [0 , fy, cy],
     [0 , 0 , 1 ]]

fig = pyplot.figure()
ax = Axes3D(fig)

# img = cv2.imread('/home/user/sample_image/image171_color.png')
# depth = np.array(Image.open('/home/user/sample_image/image171_depth.png'))

img = cv2.imread('/home/user/bag_images2/color_image0.png')
depth = np.array(Image.open('/home/user/bag_images2/depth_image0.png'))

rgb = np.reshape(img, (len(img) * len(img[0]), 3))
rgb = rgb.astype("float32")
rgb = rgb / 255

# cam_scale *= 0.75

depthInpaint = depth * cam_scale
depthInpaint = depthInpaint.astype("float32")

range_x = np.arange(1, len(depth[0]) + 1)
range_y = np.arange(1, len(depth) + 1)

x, y = np.meshgrid(range_x, range_y)

x3 = (x - cx) * depthInpaint * 1 / fx
y3 = (y - cy) * depthInpaint * 1 / fy
z3 = depthInpaint

x3 = np.reshape(x3, len(x3) * len(x3[0]))
y3 = np.reshape(y3, len(y3) * len(y3[0]))
z3 = np.reshape(z3, len(z3) * len(z3[0]))

pointsMat = np.vstack((x3, z3, -y3))

print("type : " + str(type(x3)))
print(x3.shape)
"""
    Random sampling.
    260631 --> 10000.
"""
sample_size = np.random.randint(len(pointsMat[0]), size=20000)
x3 = pointsMat[0, sample_size]
y3 = pointsMat[2, sample_size]
z3 = pointsMat[1, sample_size]
rgb = rgb[sample_size, :]

bgr = np.zeros((len(rgb), 3))
bgr[:, 0] = rgb[:, 2]
bgr[:, 1] = rgb[:, 1]
bgr[:, 2] = rgb[:, 0]

# x3 = np.append(x3, 1.0)
# y3 = np.append(y3, 1.0)
# z3 = np.append(z3, 1.0)
#
# x3 = np.append(x3, -1.0)
# y3 = np.append(y3, 1.0)
# z3 = np.append(z3, 1.0)
#
# x3 = np.append(x3, 1.0)
# y3 = np.append(y3, -1.0)
# z3 = np.append(z3, 1.0)
#
# x3 = np.append(x3, -1.0)
# y3 = np.append(y3, -1.0)
# z3 = np.append(z3, 1.0)
#
#
# x3 = np.append(x3, 1.0)
# y3 = np.append(y3, 1.0)
# z3 = np.append(z3, 0.0)
#
# x3 = np.append(x3, -1.0)
# y3 = np.append(y3, 1.0)
# z3 = np.append(z3, 0.0)
#
# x3 = np.append(x3, 1.0)
# y3 = np.append(y3, -1.0)
# z3 = np.append(z3, 0.0)
#
# x3 = np.append(x3, -1.0)
# y3 = np.append(y3, -1.0)
# z3 = np.append(z3, 0.0)
ax.scatter(x3, z3, y3, s=1, c=bgr, depthshade=False)
pyplot.show()
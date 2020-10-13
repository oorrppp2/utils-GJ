from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys
mat_file = io.loadmat('/media/user/ssd_1TB/YCB_dataset/data_syn3/030000-meta.mat')
# mat_file = io.loadmat('/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_dataset/data/0000/000001-meta.mat')
# mat_file = io.loadmat('/home/user/PycharmProjects/DenseFusion/experiments/eval_result/ycb/Densefusion_iterative_result/0000.mat')
# mat_file = io.loadmat('/home/user/PycharmProjects/DenseFusion/YCB_Video_toolbox/results_PoseCNN_RSS2018/000000.mat')
while 1:
    try:
        data = mat_file.popitem()
    except:
        break
    # print(str(data['vertmap']))
    # print(type(data))
    print("--------------------")
    print(str(data))

# color = cv2.imread('/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_dataset/data/0000/000001-color.png')
# depth = cv2.imread('/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_dataset/data/0000/000001-depth.png')
# label = cv2.imread('/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_dataset/data/0000/000001-label.png')
# # print(label)
# print("shape", str(label.shape))
# print("row : " + str(len(label)))
# print("col : " + str(len(label[0])))
# for i in range(len(label)):
#     for j in range(len(label[0])):
#         for k in label[i][j]:
#             if k != 0:
#                 print(k)

# cv2.imshow("color", color)
# cv2.imshow("depth", depth)
# cv2.imshow("label", label)
# cv2.waitKey(0)
# s = 'aa/012345'
# print(s)
# s = s[:2] + '1' + s[2:]
# print(s)
# # print(s[-6:])
from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys
import scipy.io as scio
import numpy.linalg as nl
# mat_file = io.loadmat('/media/user/ssd_1TB/YCB_dataset/data_syn3/030000-meta.mat')
# result_mat_file = io.loadmat('/home/user/python_projects/DenseFusion/experiments/eval_result/ycb/Densefusion_iterative_result/0012.mat')
# mat_file = io.loadmat('/home/user/python_projects/DenseFusion/YCB_Video_toolbox/results_PoseCNN_RSS2018/000000.mat')
while 1:
    try:
        data = result_mat_file.popitem()
    except:
        break
    # print(str(data['vertmap']))
    # print(type(data))
    # print("--------------------")
    # print(str(data))

# print('\n ================================================ \n')
# gt_mat_file = io.loadmat('/media/user/ssd_1TB/YCB_dataset/data/0048/000013-meta.mat')
# while 1:
#     try:
#         data = gt_mat_file.popitem()
#     except:
#         break
#     # print(str(data['vertmap']))
#     # print(type(data))
#     print("--------------------")
#     print(str(data))


posecnn_meta = scio.loadmat('/media/user/ssd_1TB/YCB_dataset/data/0048/000013-meta.mat')
extrinsic_1 = np.array(posecnn_meta['rotation_translation_matrix'])
# print(posecnn_meta)

vert_map = posecnn_meta['vertmap']
cv2.imshow("vert_map", vert_map)
cv2.waitKey(0)
# print(posecnn_meta['cls_indexes'].flatten())
# print(posecnn_meta['poses'].shape)
# print(posecnn_meta['poses'][:,:,0])
# print(posecnn_meta['poses'][:,:3,0])
# print(posecnn_meta['poses'][:,:3,0].shape)
# print(posecnn_meta['poses'][:,3,0])
# print(posecnn_meta['poses'][:,3,0].shape)

# print(posecnn_meta[''])

# print(posecnn_meta['center'])
# d_extrinsic = extrinsic_2 - extrinsic_1
# print("d_extrinsic : " + str(d_extrinsic))
# d_camera_pose_2 = camera_pose_1 + np.dot(d_extrinsic[:,:3].T, d_extrinsic[:,3])
# print("camera_pose_1 + d_extrinsic : " + str(d_camera_pose_2))

# x = np.zeros((2,3))
# x = [[1,2,3],[5,6,7]]
# print(nl.inv(x))


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
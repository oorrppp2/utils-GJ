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

# zc = np.zeros((1,3))
# print(zc.shape)
# zc[0,0] = 0
# zc[0,1] = 0
# zc[0,2] = 1
# print(zc.T)
zc = np.array([0,0,1])

posecnn_meta = scio.loadmat('/media/user/ssd_1TB/YCB_dataset/data/0048/000068-meta.mat')
extrinsic_1 = np.array(posecnn_meta['rotation_translation_matrix'])
print(extrinsic_1)
R_1 = extrinsic_1[:,:3]
T_1 = extrinsic_1[:,3]
# print(extrinsic_1[:,:3])
camera_pose_1 = -np.dot(nl.inv(R_1), T_1)
print("camera_pose_1 : " + str(camera_pose_1))

# print(np.dot(extrinsic_1[:,:3], nl.inv(extrinsic_1[:,:3])))
# print(np.dot(extrinsic_1[:,:3].T, extrinsic_1[:,:3]))

posecnn_meta = scio.loadmat('/media/user/ssd_1TB/YCB_dataset/data/0048/000111-meta.mat')
extrinsic_2 = np.array(posecnn_meta['rotation_translation_matrix'])
R_2 = extrinsic_2[:,:3]
T_2 = extrinsic_2[:,3]
# extrinsic_2[2,3] += 1
print(extrinsic_2)
camera_pose_2 = -np.dot(nl.inv(R_2), T_2)
print("camera_pose_2 : " + str(camera_pose_2))

zw_1 = np.dot(nl.inv(R_1), zc.T)
print("zw_1 : " + str(zw_1))

pan_1 = np.arctan2(zw_1[1], zw_1[0]) - (np.pi / 2.0)
tilt_1 = np.arctan2(zw_1[2], np.sqrt((zw_1[0]*zw_1[0] + zw_1[1]*zw_1[1])))

pan_2 = np.arctan2(camera_pose_2[1], camera_pose_2[0]) - (np.pi / 2.0)
tilt_2 = np.arctan2(camera_pose_2[2], np.sqrt((camera_pose_2[0]**2 + camera_pose_2[1]**2)))
print("pan_1 : " + str(pan_1))
print("tilt_1 : " + str(tilt_1))
print("pan_2 : " + str(pan_2))
print("tilt_2 : " + str(tilt_2))
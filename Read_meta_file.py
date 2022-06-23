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
from PIL import Image
import scipy.io as scio
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, qinverse
from transforms3d.euler import quat2euler, mat2euler, euler2quat

dataset_root_dir = '/media/user/ssd_1TB/YCB_dataset'
dataset_config_dir = '/home/user/python_projects/DenseFusion/datasets/ycb/dataset_config'
ycb_toolbox_dir = '/home/user/python_projects/DenseFusion/YCB_Video_toolbox'
testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()

# f = scio.loadmat("/media/user/ssd_1TB/YCB_dataset/data/0040/000001-meta.mat")
# vert = f['vertmap']
# print(f)
# print(vert.shape)
# print(vert[300:340, 300:340, :])
# cv2.imshow("vert", vert)
# cv2.waitKey(0)

now = 2093
img = cv2.imread('{0}/{1}-color.png'.format(dataset_root_dir, testlist[now]))
depth = np.array(Image.open('{0}/{1}-depth.png'.format(dataset_root_dir, testlist[now])))
posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
# label = np.array(posecnn_meta['labels'])

f = scio.loadmat('{0}/{1}-meta.mat'.format(dataset_root_dir, testlist[now]))
print(posecnn_meta['rois'])
for rois in posecnn_meta['rois']:
    if rois[1] == 12:
        print(rois)
# print(f)
# print(f['poses'][:,:,0])
# print(f['poses'][:,:,1])
# print(f['poses'][:,:,2])
# print(f['poses'][:,:,3])
# print(f['poses'][:,:,4])
rot = f['poses'][:,:,1]
print(rot[:3,:3])
quat = mat2quat(rot[:3,:3])
print(quat)
euler = np.array(list(quat2euler(quat)))
degree = euler * 180 / np.pi
print(degree)

print(np.array(list(quat2euler([0.498, 0.815, 0.2852, -0.0796]))) * 180 / np.pi)
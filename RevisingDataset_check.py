from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys
import os
import random
import shutil

mat_file = io.loadmat('/home/user/SUNRGBD/SUNRGBDMeta.mat')

r = open('/home/user/darknet-pose/data/pose_yolo_valid_v2.txt', mode='rt', encoding='utf-8')

splitlinestr = str.splitlines(r.read())
nums = []
# path = "/home/user/darknet/data/pose_yolo/"
# print(splitlinestr)
for str_line in splitlinestr:
    txt_path = "/home/user/darknet-pose/" + str_line[:-5] + ".txt"

    f = open(txt_path, mode='rt')
    splitf = str.splitlines(f.read())

    nums.append(txt_path[48:-4])
    pass
r.close()

for range_index in range(4):
    data = mat_file.popitem()
    if (data[0] == 'SUNRGBDMeta'):
        for n in nums:
            image_index = int(n)
            print("Img path : ", data[1][0][image_index][5][0])
            color_raw = cv2.imread(data[1][0][image_index][5][0], cv2.COLOR_RGB2BGR)
            cv2.imwrite("/home/user/darknet-pose/data/pose_yolo_3c_valid/"+str(image_index)+".jpg", color_raw)
            # break
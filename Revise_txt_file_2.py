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

r = open('/home/user/python_projects/DenseFusion/datasets/ycb/dataset_config/test_data_list.txt', mode='rt', encoding='utf-8')
save_txt = ""
splitlinestr = str.splitlines(r.read())
cnt = 0
# pose_yolo_train_v2.txt 파일을 읽어와서 한 줄씩 파일을 load.
for str_line in splitlinestr:
    if cnt % 10 == 0:
        save_txt += str_line + '\n'
        # print(str_line)
    cnt += 1



    # ff = open(txt_path, mode='wt')
    # ff.write(save_txt)
    # ff.close()
    pass

r.close()

print(save_txt)


ff = open('/home/user/python_projects/DenseFusion/datasets/ycb/dataset_config/small_test_data_list.txt', mode='wt')
ff.write(save_txt)
ff.close()
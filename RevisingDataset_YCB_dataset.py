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

r = open('/home/user/python_projects/DenseFusion/train_data_list.txt', mode='rt', encoding='utf-8')
save_path_txt = ""
cnt = 0
empty_cnt = 0
splitlinestr = str.splitlines(r.read())
save_txt = ""
for str_line in splitlinestr:
    if int(str_line[-6:]) % 4 == 0:
        # print(int(str_line[-6:]))
        cnt += 1
        save_txt += str_line+"\n"
print("cnt : " + str(cnt))

ff = open('/home/user/python_projects/DenseFusion/train_data_list_revised.txt', mode='wt')
ff.write(save_txt)
ff.close()

r.close()
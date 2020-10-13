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

r = open('/home/user/darknet-master/data/python.list', mode='rt', encoding='utf-8')
save_path_txt = ""
splitlinestr = str.splitlines(r.read())

# pose_yolo_train_v2.txt 파일을 읽어와서 한 줄씩 파일을 load.
imgs = []
for str_line in splitlinestr:
    txt_path = "/home/user/darknet-master/" + str_line[:-4] + ".txt"
    print(txt_path)
    f = open(txt_path, mode='rt')
    f_line = str.splitlines(f.read())
    txt_v5 = ""

    save_txt = ""
    for line in f_line:
        print(line[:1])
        # theta = math.atan2(t_im_revised*2-1, t_re_revised*2-1)
        # theta =math.degrees(theta)
        # save_txt += line[:38] + (str(format(t_re_revised, '.6f')) + "000000")[:8] + " " + (str(format(t_im_revised, '.6f')) + "000000")[:8] + "\n"
        pass
    f.close()

    pass
r.close()

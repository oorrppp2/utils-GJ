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

r = open('/home/user/darknet-pose/data/pose_yolo_valid_v6.txt', mode='rt', encoding='utf-8')
save_path_txt = ""
splitlinestr = str.splitlines(r.read())

# pose_yolo_train_v2.txt 파일을 읽어와서 한 줄씩 파일을 load.
for str_line in splitlinestr:

    txt_path = "/home/user/darknet-pose/" + str_line[:-5] + ".txt"
    print(txt_path)
    f = open(txt_path, mode='rt')
    f_line = str.splitlines(f.read())
    txt_v5 = ""

    save_txt = ""
    for line in f_line:
        t_re = float(line[38:46])
        t_im = float(line[47:])
        # theta = math.atan2(t_im, t_re)
        t_re_revised = (t_re + 1) / 2.0
        t_im_revised = (t_im + 1) / 2.0
        # theta = math.atan2(t_im_revised*2-1, t_re_revised*2-1)
        # theta =math.degrees(theta)
        save_txt += line[:38] + (str(format(t_re_revised, '.6f')) + "000000")[:8] + " " + (str(format(t_im_revised, '.6f')) + "000000")[:8] + "\n"
        pass
    f.close()

    print(save_txt)

    # line에 "12" = desk 가 있으면 넘어가고, desk가 아니면 save_txt에 한 줄씩 추가.
    # for line in splitf:
    #     if line[:2] == "12":
    #         pass
    #     else:
    #         save_txt += line+"\n"
    #     pass

    ff = open(txt_path, mode='wt')
    ff.write(save_txt)
    ff.close()
    pass

r.close()

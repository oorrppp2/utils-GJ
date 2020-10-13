from scipy import io
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys

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

r = open('/home/user/darknet-pose/data/pose_yolo_train_v3.txt', mode='rt', encoding='utf-8')
save_path_txt = ""
splitlinestr = str.splitlines(r.read())

histo_path = "/home/user/histo_result.jpg"
histo_arr = np.zeros(360)
# pose_yolo_train_v2.txt 파일을 읽어와서 한 줄씩 파일을 load.
for str_line in splitlinestr:

    txt_path = "/home/user/darknet-pose/" + str_line[:-5] + ".txt"
    # print(txt_path)
    f = open(txt_path, mode='rt')
    f_line = str.splitlines(f.read())
    txt_v5 = ""

    for line in f_line:
        t_re = float(line[38:46])*2-1
        t_im = float(line[47:])*2-1
        theta = np.round(math.degrees(math.atan2(t_im, t_re)))
        if theta == 180:
            theta = 179
            pass
        index = int(theta+180)
        histo_arr[index] += 1
        pass
    f.close()
    pass

r.close()

x2 = np.arange(0, 360)
y2 = histo_arr

plt2, = plt.plot(x2, y2, 'or', label='Histo')
legend_all = plt.legend(handles=[plt2], loc='upper right')
# art_legend = plt.g
plt.show()

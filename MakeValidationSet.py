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

r = open('/home/user/darknet/data/pose_yolo.txt', mode='rt', encoding='utf-8')

splitlinestr = str.splitlines(r.read())
path = "/home/user/darknet/data/pose_yolo/"
arr = []
# for i in range(9509):
for i in range(10):
    str_line = str(splitlinestr[i])
    n = str_line[34:-5]
    arr.append(n)
# print(arr[-1])

# os.remove(path+"test.txt")

train_txt = ""

for i in range(1901):
    s = random.choice(arr)
    img_s = path+s+".tiff"
    txt_s = path+s+".txt"
    shutil.move(img_s, "/home/user/darknet/data/pose_yolo_valid/"+s+".tiff")
    shutil.move(txt_s, "/home/user/darknet/data/pose_yolo_valid/"+s+".txt")
    train_txt += img_s+"\n"
    # print(img_s)

file = open("/home/user/darknet/data/pose_yolo_valid.txt", mode='wt')
file.write(train_txt)
file.close()
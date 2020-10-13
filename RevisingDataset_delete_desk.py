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

r = open('/home/user/darknet-pose/data/pose_yolo_train_v2.txt', mode='rt', encoding='utf-8')
save_path_txt = ""
cnt = 0
splitlinestr = str.splitlines(r.read())

# pose_yolo_train_v2.txt 파일을 읽어와서 한 줄씩 파일을 load.
for str_line in splitlinestr:
    txt_path = "/home/user/darknet-pose" + str_line[:-5] + ".txt"
    save_txt = ""

    f = open(txt_path, mode='rt')

    # dataset이 저장된 text파일을 읽어와서 한 줄씩 split
    splitf = str.splitlines(f.read())

    # print("*" + txt_path[48:-4] + "*")

    # line에 "12" = desk 가 있으면 넘어가고, desk가 아니면 save_txt에 한 줄씩 추가.
    for line in splitf:
        if line[:2] == "12":
            pass
        else:
            save_txt += line+"\n"
        pass
    f.close()

    # 만약 save_txt가 비었다는 것은 아무 label도 없다는 것이므로
    # 지워야함.
    img_path = txt_path[:-3] + "tiff"
    if save_txt != "":
        cnt += 1
        save_path_txt += img_path[23:46]+"5/"+img_path[48:]+"\n"
        # print(img_path[23:46]+"5/"+img_path[48:])
        # os.remove(txt_path)
        # os.remove(img_path)
        # print(txt_path)
        # print(img_path)
        # print(txt_path[48:-4])

    # print(txt_path)
    ff = open(txt_path, mode='wt')
    ff.write(save_txt)
    ff.close()
    pass

r.close()

print(cnt)

#     n = str_line[34:-5]

#     arr.append(n)
# print(arr[-1])

# os.remove(path+"test.txt")


# for i in range(1901):
#     s = random.choice(arr)
#     img_s = path+s+".tiff"
#     txt_s = path+s+".txt"
#     shutil.move(img_s, "/home/user/darknet/data/pose_yolo_valid/"+s+".tiff")
#     shutil.move(txt_s, "/home/user/darknet/data/pose_yolo_valid/"+s+".txt")
#     train_txt += img_s+"\n"
#     # print(img_s)
#
# file = open("/home/user/darknet/data/pose_yolo_valid.txt", mode='wt')
# file.write(train_txt)
# file.close()
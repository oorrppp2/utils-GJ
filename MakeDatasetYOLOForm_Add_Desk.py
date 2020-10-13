from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys
import shutil

# np.set_printoptions(threshold=sys.maxsize)
r = open('/home/user/darknet-pose/data/pose_yolo_v5.txt', mode='rt', encoding='utf-8')
save_path_txt = ""
cnt = 0
splitlinestr = str.splitlines(r.read())
for str_line in splitlinestr:
    txt_path = "/home/user/darknet-pose/data/pose_yolo_train_v3/" + str_line[24:-5] + ".txt"
    txt_path_valid = "/home/user/darknet-pose/data/pose_yolo_valid_v3/" + str_line[24:-5] + ".txt"
    txt_path_v5 = "/home/user/darknet-pose/" + str_line[:-5] + ".txt"
    # print(txt_path)
    # print(txt_path_valid)
    print(txt_path_v5)
    # print()
    save_txt = ""

    f_v5 = open(txt_path_v5, mode='rt')
    f_v5_line = str.splitlines(f_v5.read())
    txt_v5 = ""
    for line in f_v5_line:
        txt_v5 += "11 " + line[2:] + "\n"
        pass
    f_v5.close()

    print(txt_v5[:-1])

    try:
        # 만약 train_v3에 있으면
        f = open(txt_path, mode='rt')
        save_txt = f.read()
        f.close()

        save_txt += txt_v5

        print("save train : \n", save_txt)
    except:
        try:
            f = open(txt_path_valid, mode='rt')
            save_txt_valid = f.read()
            f.close()
            save_txt_valid += txt_v5

            wf = open(txt_path_valid, mode="wt")
            wf.write(save_txt_valid)
            wf.close()

            print("save valid: ", save_txt_valid)
            pass
        except:

            # new_path = txt_path_v5[24:]
            # print(" * /home/user/darknet-pose/data/imsi/" + str_line[24:-5] + ".txt")

            # wf = open("/home/user/darknet-pose/data/imsi/" + str_line[24:-5] + ".txt", mode='wt')
            # wf.write(txt_v5)
            # wf.close()
            # img_path = txt_path_v5[:-4] + ".tiff"
            # dst_path = "/home/user/darknet-pose/data/imsi/" + str_line[24:-5] + ".tiff"
            # shutil.copy2(img_path, dst_path)
            cnt += 1
            # print(img_path)
            pass
        pass
    pass

r.close()

print(cnt)

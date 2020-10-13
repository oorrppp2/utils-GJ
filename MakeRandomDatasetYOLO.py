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
new_path = []
splitlinestr = str.splitlines(r.read())
for str_line in splitlinestr:
    txt_path = "/home/user/darknet-pose/data/pose_yolo_train_v3/" + str_line[24:-5] + ".txt"
    txt_path_valid = "/home/user/darknet-pose/data/pose_yolo_valid_v3/" + str_line[24:-5] + ".txt"
    txt_path_v5 = "/home/user/darknet-pose/" + str_line[:-5] + ".txt"
    # print(txt_path)
    # print(txt_path_valid)
    # print(txt_path_v5)
    # print()
    save_txt = ""
    f_v5 = open(txt_path_v5, mode='rt')
    f_v5_line = str.splitlines(f_v5.read())
    txt_v5 = ""
    for line in f_v5_line:
        txt_v5 += "11 " + line[2:] + "\n"
        pass
    f_v5.close()

    # print(txt_v5[:-1])

    try:
        # 만약 train_v3에 있으면
        f = open(txt_path, mode='rt')
        save_txt = f.read()
        f.close()

        save_txt += txt_v5

        # print("save train : \n", save_txt)
    except:
        try:
            f = open(txt_path_valid, mode='rt')
            save_txt_valid = f.read()
            f.close()
            save_txt_valid += txt_v5

            # print("save valid: ", save_txt_valid)
            pass
        except:

            new_path.append("/home/user/darknet-pose/data/imsi/" + str_line[24:-5] + ".tiff")
            # print(" * /home/user/darknet-pose/data/imsi/" + str_line[24:-5] + ".txt")

            # wf = open("/home/user/darknet-pose/data/imsi/" + str_line[24:-5] + ".txt", mode='wt')
            # wf.write(txt_v5)
            # wf.close()
            # img_path = "/home/user/darknet-pose/data/imsi/" + str_line[24:-5] + ".tiff"
            # dst_path = "/home/user/darknet-pose/data/imsi_train/" + str_line[24:-5] + ".tiff"
            # shutil.copy2(img_path, dst_path)
            cnt += 1
            # print(img_path)
            pass
        pass
    pass

r.close()

# print(cnt)
# print(new_path)

random_arr = np.random.binomial(n=1, p=0.27, size=10335)
max_size = 0
valid_path = ""
train_path = ""
for line in new_path:
    # print(line[34:-5])
    if max_size < 53 and random_arr[int(line[34:-5])] == 1:
        # print("valid : ", line[:33] + "_valid/" + line[34:])
        valid_src_path = line
        valid_dst_path = line[:33] + "_valid/" + line[34:]
        valid_path += valid_dst_path + "\n"

        valid_txt_src_path = line[:-4]+"txt"
        valid_txt_dst_path = valid_dst_path[:-4]+"txt"

        shutil.copy2(valid_src_path, valid_dst_path)
        shutil.copy2(valid_txt_src_path, valid_txt_dst_path)
        # print(txt_src_path)
        # print(txt_dst_path)
        # print(src_path)
        # print(dst_path)
        max_size += 1
        pass
    else:
        src_path = line
        dst_path = line[:33] + "_train/" + line[34:]
        train_path += dst_path + "\n"
        txt_src_path = line[:-4]+"txt"
        txt_dst_path = dst_path[:-4]+"txt"
        shutil.copy2(src_path, dst_path)
        shutil.copy2(txt_src_path, txt_dst_path)
        # print("train : ", line[:33] + "_valid/" + line[34:])
        pass
    pass
print("train")
print(train_path)
print("valid")
print(valid_path)
print(max_size)

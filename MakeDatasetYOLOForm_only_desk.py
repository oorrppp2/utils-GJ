from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys

# np.set_printoptions(threshold=sys.maxsize)

mat_file = io.loadmat('/home/user/SUNRGBD/SUNRGBDMeta.mat')
path_tail = "/home/user/darknet-pose/data/"
max_size = 0

for range_index in range(4):

    data = mat_file.popitem()
    if (data[0] == 'SUNRGBDMeta'):
        sample_size = np.arange(10335)
        random_arr = np.random.binomial(n=1, p=0.25, size=10335)
        classes_txt = ""
        train_txt = ""
        valid_txt = ""

        train_s = path_tail + "pose_yolo_train_v5.txt"
        valid_s = path_tail + "pose_yolo_valid_v5.txt"

        index = 0
        pers = 0

        imgs = 0
        imgs_valid = 0
        imgs_train = 0

        for ele in sample_size:
            image_index = ele
            if (data[1][0][image_index][1].size < 1):
                continue
                pass
            if image_index % 1000 == 0 and pers < 100:
                pers+=10
                print(pers,"% 진행")

            img_s = path_tail+"pose_yolo_train_v5/" + str(ele) + ".tiff"
            img_s_valid = path_tail+"pose_yolo_valid_v5/" + str(ele) + ".tiff"

            txt_s = path_tail+"pose_yolo_train_v5/" + str(ele) + ".txt"
            txt_s_valid = path_tail+"pose_yolo_valid_v5/" + str(ele) + ".txt"

            img_s_v6 = path_tail+"pose_yolo_train_v6/" + str(ele) + ".tiff"
            img_s_valid_v6 = path_tail+"pose_yolo_valid_v6/" + str(ele) + ".tiff"

            txt_s_v6 = path_tail+"pose_yolo_train_v6/" + str(ele) + ".txt"
            txt_s_valid_v6 = path_tail+"pose_yolo_valid_v6/" + str(ele) + ".txt"

            color_raw = cv2.imread(data[1][0][image_index][5][0], -1)
            depth_raw = cv2.imread(data[1][0][image_index][4][0], -1)
            row = len(color_raw)
            col = len(color_raw[0])

            txt = ""
            txt_v6 = ""
            for groundtruth3DBB in data[1][0][image_index][1]:
                for items in groundtruth3DBB:

                    if (items[7].size < 1):
                        continue

                    """"check"""
                    label = items[3][0]
                    if label == "desk":
                        pass
                    else:
                        continue
                        pass

                    """ 2D bounding box """
                    gt2DBB = items[7][0]
                    bx = (items[7][0][0] + items[7][0][2] / 2) / col
                    by = (items[7][0][1] + items[7][0][3] / 2) / row
                    bw = items[7][0][2] / col
                    bh = items[7][0][3] / row
                    orientation = items[6][0]
                    t_re = (orientation[0] + 1.0) / 2.0
                    t_im = (orientation[1] + 1.0) / 2.0

                    t_re_ori = orientation[0]
                    t_im_ori = orientation[1]

                    t_re_v6 = orientation[0]
                    t_im_v6 = orientation[1]

                    if t_re_ori * t_im_ori > 0:
                        t_re_v6 = abs(t_re_ori)
                        t_im_v6 = abs(t_im_ori)
                        pass
                    elif t_re_ori * t_im_ori < 0:
                        t_re_v6 = abs(t_im_ori)
                        t_im_v6 = abs(t_re_ori)
                        pass
                    else:
                        t_re_v6 = 0
                        t_im_v6 = 1
                        pass

                    # print("t_re_ori : ", t_re_ori)
                    # print("t_im_ori : ", t_im_ori)
                    # print("t_re_v6 : ", t_re_v6)
                    # print("t_im_v6 : ", t_im_v6)
                    # print("=======================================")

                    txt += "0"
                    txt += " " + (str(format(bx, '.6f')) + "000000")[:8]
                    txt += " " + (str(format(by, '.6f')) + "000000")[:8]
                    txt += " " + (str(format(bw, '.6f')) + "000000")[:8]
                    txt += " " + (str(format(bh, '.6f')) + "000000")[:8]
                    txt += " " + (str(format(t_re, '.6f')) + "000000")[:8]
                    txt += " " + (str(format(t_im, '.6f')) + "000000")[:8] + "\n"

                    txt_v6 += "0"
                    txt_v6 += " " + (str(format(bx, '.6f')) + "000000")[:8]
                    txt_v6 += " " + (str(format(by, '.6f')) + "000000")[:8]
                    txt_v6 += " " + (str(format(bw, '.6f')) + "000000")[:8]
                    txt_v6 += " " + (str(format(bh, '.6f')) + "000000")[:8]
                    txt_v6 += " " + (str(format(t_re_v6, '.6f')) + "000000")[:8]
                    txt_v6 += " " + (str(format(t_im_v6, '.6f')) + "000000")[:8] + "\n"

                    pass

                pass

            if txt != "":
                if max_size < 334 and random_arr[image_index] == 1:
                    max_size += 1
                    f = open(txt_s_valid, mode='wt')
                    f.write(txt)
                    f.close()

                    ff = open(txt_s_valid_v6, mode='wt')
                    ff.write(txt_v6)
                    ff.close()

                    depthInpaint = depth_raw / 65535 * 255
                    depthInpaint = depthInpaint.astype("uint8")

                    transformed = np.dstack([color_raw, depthInpaint])
                    # print(img_s_valid)
                    cv2.imwrite(img_s_valid, transformed)
                    valid_txt += "data/pose_yolo_valid_v5/" + str(ele) + ".tiff" + "\n"
                    pass
                else:
                    f = open(txt_s, mode='wt')
                    f.write(txt)
                    f.close()

                    ff = open(txt_s_v6, mode='wt')
                    ff.write(txt_v6)
                    ff.close()

                    depthInpaint = depth_raw / 65535 * 255
                    depthInpaint = depthInpaint.astype("uint8")

                    transformed = np.dstack([color_raw, depthInpaint])
                    # print(img_s)
                    cv2.imwrite(img_s, transformed)
                    train_txt += "data/pose_yolo_train_v5/" + str(ele) + ".tiff" + "\n"
                    pass
                # print(txt)
                # print("==================================")
                pass
            else:
                continue
                pass

            imgs += 1

            if random_arr[image_index] == 1:
                imgs_valid += 1
                pass
            else:
                imgs_train += 1
                pass
            if max_size == 334:
                print("last index : ", image_index)
                max_size += 1
                pass

            pass

        tf = open(train_s, mode='wt')
        tf.write(train_txt)
        tf.close()

        tf = open(valid_s, mode='wt')
        tf.write(valid_txt)
        tf.close()

        # print("end")
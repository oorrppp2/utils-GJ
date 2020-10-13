from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys


def flip_towards_viewer(normals, points):
    mat = np.matlib.repmat(np.sqrt(np.sum(points * points, 1)), 3, 1)
    points = points / mat
    # print(points)
    proj = np.sum(points * normals, 1)
    flip = proj > 0
    normals[flip, :] = -normals[flip, :]
    return normals


# np.set_printoptions(threshold=sys.maxsize)

skip_img = [50,
51,
55,
115,
116,
123,
125,
155,
246,
285,
306,
388,
409,
679,
715,
716,
739,
778,
802,
818,
852,
918,
976,
1115,
1131,
1354,
1365,
1677,
1679,
1693,
1694,
1695,
1696,
1699,
1705,
1861,
1879,
1882,
1884,
1898,
1900,
1915,
1948,
2042,
2060,
2072,
2087,
2092,
2093,
2112,
2113,
2130,
2141,
2176,
2180,
2182,
2186,
2188,
2194,
2212,
2214,
2230,
2232,
2237,
2238,
2240,
2303,
2304,
2342,
2343,
2393,
2418,
2428,
2449,
2498,
2509,
2619,
2802,
2852,
2871,
2919,
2933,
2977,
3133,
3223,
3265,
3283,
3284,
3300,
3301,
3323,
3335,
3432,
3441,
3456,
3487,
3488,
3490,
3493,
3497,
3514,
3543,
3545,
3614,
3615,
3625,
3700,
3796,
3808,
3902,
3970,
4004,
4017,
4019,
4026,
4037,
4047,
4052,
4053,
4112,
4113,
4138,
4167,
4195,
4219,
4254,
4266,
4289,
4312,
4376,
4378,
4439,
4522,
4564,
4628,
4639,
4642,
4699,
4847,
4924,
5022,
5090,
5091,
5096,
5102,
5120,
5204,
5216,
5530,
5357,
5377,
5449,
5756,
5808,
5809,
5864,
5869,
5877,
5886,
5942,
5994,
6030,
6125,
6148,
6171,
6200,
6214,
6215,
6216,
6217,
6219,
6230,
6238,
6240,
6243,
6245,
6248,
6250,
6252,
6253,
6257,
6258,
6260,
6265,
6267,
6272,
6300,
6335,
6388,
6410,
6420,
6451,
6461,
6465,
6495,
6501,
6506,
6517,
6533,
6555,
6621,
6654,
6683,
6697,
6753,
6802,
6805,
6809,
6908,
6941,
6945,
6989,
6995,
7002,
7010,
7027,
7028,
7029,
7049,
7054,
7056,
7057,
7069,
7082,
7086,
7122,
7166,
7173,
7175,
7181,
7183,
7202,
7208,
7223,
7225,
7231,
7232,
7238,
7240,
7244,
7245,
7247,
7259,
7261,
7275,
7288,
7292,
7295,
7308,
7310,
7311,
7312,
7313,
7320,
7324,
7340,
7341,
7343,
7352,
7359,
7385,
7386,
7408,
7410,
7413,
7418,
7421,
7423,
7430,
7431,
7438,
7440,
7443,
7445,
7448,
7450,
7451,
7454,
7457,
7458,
7460,
7462,
7463,
7464,
7468,
7472,
7480,
7487,
7488,
7491,
7495,
7507,
7525,
7535,
7546,
7566,
7585,
7588,
7590,
7607,
7615,
7617,
7622,
7623,
7651,
7653,
7655,
7656,
7663,
7665,
7667,
7676,
7689,
7692,
7704,
7708,
7711,
7742,
7760,
7765,
7782,
7883,
7889,
7933,
7966,
7973,
7976,
8010,
8013,
8027,
8029,
8050,
8062,
8118,
8128,
8132,
8199,
8225,
8229,
8231,
8322,
8334,
8403,
8438,
8446,
8493,
8501,
8528,
8532,
8537,
8538,
8563,
8564,
8568,
8569,
8570,
8571,
8589,
8624,
8635,
8659,
8678,
8698,
8703,
8710,
8716,
8744,
8795,
8805,
8808,
8857,
8858,
8890,
8900,
8927,
8928,
8958,
8996,
9006,
9026,
9031,
9037,
9041,
9056,
9061,
9069,
9109,
9110,
9120,
9221,
9236,
9238,
9269,
9270,
9275,
9279,
9285,
9315,
9324,
9345,
9355,
9358,
9382,
9408,
9567,
9588,
9599,
9641,
9642,
9706,
9711,
9716,
9717,
9749,
9808,
9856,
9857,
9903,
9922,
9926,
9942,
9955,
9960,
9963,
9981,
9999,
10009,
10078,
10082,
10114,
10153,
5,
23,
137,
682,
755,
902,
938,
1009,
1012,
1023,
1075,
1111,
1143,
1162,
1163,
1612,
1704,
1707,
1789,
1830,
1866,
1903,
1907,
1913,
1933,
2023,
2030,
2037,
2078,
2085,
2088,
2187,
2208,
2213,
2223,
2229,
2231,
2352,
2392,
2409,
2427,
2506,
2621,
2622,
2679,
3007,
3294,
3338,
3389,
3425,
3449,
3455,
3459,
3483,
3547,
3611,
3809,
3961,
3980,
4005,
4044,
4067,
4130,
4137,
4228,
4229,
4238,
4249,
4293,
4320,
4397,
4434,
4448,
4481,
4497,
4503,
4528,
4566,
4620,
4773,
4881,
5469,
5507,
5614,
5697,
5821,
5894,
6133,
6213,
6221,
6262,
6264,
6269,
6293,
6346,
6351,
6384,
6483,
6538,
6542,
6597,
6673,
6675,
6758,
6815,
]
mat_file = io.loadmat('/home/user/SUNRGBD/SUNRGBDMeta.mat')

path_tail = "/home/user/darknet-pose/data/"
# path_tail = "/home/user/"

ID = {"bed": 0, "drawer": 1, "chair": 2,
      "counter": 3, "door": 4, "painting": 5, "pillow": 6,
      "shelf": 7, "sofa": 8, "toilet": 9, "tv": 10, "table": 11, "desk": 12}
ID_cnt_train = np.zeros(13)
ID_cnt_valid = np.zeros(13)
# print("ID_cnt :", ID_cnt)

classes = [
    "bed",
    "cabinet",
    "drawer",
    "kitchen_cabinet",
    "file_cabinet",
    "night_stand",
    "chair",
    "kitchen_counter",
    "counter",
    "door",
    "pillow",
    "painting",
    "picture",
    "shelf",
    "bookshelf",
    "sofa_chair",
    "sofa",
    "toilet",
    "tv",
    "table",
    "endtable",
    "coffee_table",
    "dining_table",
    "desk",
]

max_size = 0
# print("size : ", len(classes))

for range_index in range(4):

    data = mat_file.popitem()
    if (data[0] == 'SUNRGBDMeta'):

        # count_num = np.zeros(1146)
        sample_size = np.arange(10335)
        random_arr = np.random.binomial(n=1, p=0.3, size=10335)
        # sample_size = np.arange(10)

        # print(count_num)
        # classes = {}
        classes_txt = ""
        train_txt = ""
        valid_txt = ""
        train_s = path_tail + "/pose_yolo_coffeetable.txt"
        index = 0

        pers = 0

        for ele in sample_size:
            # image_index = 0
            # print(ele, "번 째")
            image_index = ele
            if image_index in skip_img :
                continue
                pass
            if (data[1][0][image_index][1].size < 1):
                continue
                pass
            if image_index % 1000 == 0 and pers < 100:
                pers+=10
                print(pers,"% 진행")

            img_s = path_tail+"pose_yolo_coffeetable/" + str(ele) + ".tiff"

            txt_s = path_tail+"pose_yolo_coffeetable/" + str(ele) + ".txt"

            color_raw = cv2.imread(data[1][0][image_index][5][0], -1)
            depth_raw = cv2.imread(data[1][0][image_index][4][0], -1)
            row = len(color_raw)
            col = len(color_raw[0])

            txt = ""
            for groundtruth3DBB in data[1][0][image_index][1]:
                for items in groundtruth3DBB:

                    if (items[7].size < 1):
                        continue

                    """"check"""
                    label = items[3][0]
                    if label == "coffee_table":
                        label = "table"
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
                    t_re = orientation[0]
                    t_im = orientation[1]

                    t_re = (t_re+1)/2
                    t_im = (t_im+1)/2

                    """
                        -1 <= t_re, t_im <= 1
                            ㄴ> 0 ~ 1
                        so, {(t_re, t_im) + 1} / 2
                    """

                    if t_re < 0 or t_re > 1 or t_im < 0 or t_im > 1 or bx < 0 or bx > 1 or by < 0 or by > 1\
                            or bw < 0 or bw > 1 or bh < 0 or bh > 1:
                        while(1):
                            print("range incorrect")
                        break

                    txt += str(ID[label])
                    txt += " " + (str(bx) + "000000")[:8]
                    txt += " " + (str(by) + "000000")[:8]
                    txt += " " + (str(bw) + "000000")[:8]
                    txt += " " + (str(bh) + "000000")[:8]
                    txt += " " + (str(t_re) + "000000")[:8]
                    txt += " " + (str(t_im) + "000000")[:8] + "\n"

                    # if max_size < 1901 and random_arr[image_index] == 1:
                    #     ID_cnt_valid[ID[label]] += 1
                    #     pass
                    # else:
                    #     ID_cnt_train[ID[label]] += 1
                    #     pass
                    pass

                pass

            # if max_size == 1881:
            #     print("last index : ", image_index)
            #     max_size += 1
            #     pass

            if txt != "":
                print("index : ", image_index)
                max_size += 1
                f = open(txt_s, mode='wt')
                f.write(txt)
                f.close()

                depthInpaint = depth_raw / 65535 * 255
                depthInpaint = depthInpaint.astype("uint8")

                transformed = np.dstack([color_raw, depthInpaint])
                # print(img_s_valid)
                cv2.imwrite(img_s, transformed)
                train_txt += "data/pose_yolo_coffeetable/" + str(ele) + ".tiff" + "\n"
                pass
            pass

        tf = open(train_s, mode='wt')
        tf.write(train_txt)
        tf.close()

        print("total : ", max_size)
        print("end")
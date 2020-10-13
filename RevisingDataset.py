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

arr = [
1181,
1184,
1185,
1188,
1189,
1190,
1206,
1214,
1222,
1236,
1243,
1246,
1247,
1248,
1249,
1266,
1269,
1272,
1276,
1277,
1284,
1285,
1286,
1287,
1291,
1294,
1297,
1299,
1304,
1306,
1316,
1320,
1321,
1322,
1323,
1325,
1326,
1328,
1331,
1345,
1375,
1408,
1429,
1443,
1448,
1449,
1451,
1457,
1458,
1460,
1461,
1467,
1469,
1475,
1478,
1481,
1483,
1487,
1494,
1497,
1511,
1516,
1534,
1570,
1574,
1575,
1576,
1577,
1579,
1581,
1607,
1613,
1619,
1625,
1626,
1634,
1637,
1639,
1641,
1651,
1655,
1656,
1667,
1692,
1742,
1809,
1810,
1814,
1816,
1825,
1854,
1856,
1872,
1874,
1952,
1975,
1977,
1979,
1993,
2009,
2020,
2052,
2111,
2116,
2323,
2386,
2514,
2522,
2525,
2528,
2532,
2535,
2538,
2623,
2628,
2996,
3001,
3086,
3087,
3109,
3121,
3124,
3131,
3132,
3134,
3135,
3145,
3318,
3375,
3742,
3826,
3895,
3987,
3988,
4353,
4355,
4405,
4406,
4407,
4408,
4409,
4541,
4542,
4543,
4575,
4591,
4601,
4605,
4607,
4722,
4750,
4804,
4890,
5008,
5146,
5169,
5178,
5196,
5308,
5324,
5360,
5363,
5399,
5402,
5455,
5495,
5591,
5623,
5763,
5768,
5772,
5778,
5779,
5780,
5883,
6078
]
r = open('/home/user/darknet-pose/data/pose_yolo_valid_v4.txt', mode='rt', encoding='utf-8')

splitlinestr = str.splitlines(r.read())
# path = "/home/user/darknet/data/pose_yolo/"
# print(splitlinestr)
for str_line in splitlinestr:
    txt_path = "/home/user/darknet-pose/" + str_line[:-5] + ".txt"
    # print(txt_path)

    # print(int(txt_path[48:-4]))
    save_txt = ""
    if int(txt_path[48:-4]) in arr:
        # print(txt_path[48:-4])
        f = open(txt_path, mode='rt')

        splitf = str.splitlines(f.read())
        flag = False

        for line in splitf:
            if line[:2] == "12":
                if flag is False:
                    print("*" + txt_path[48:-4] + "*")
                    flag = True
                    pass
                # print(line)
                pass
            else:
                save_txt += line+"\n"
            # print(line)
            pass
        f.close()

        ff = open(txt_path, mode='wt')
        ff.write(save_txt)
        print(save_txt)
        # if save_txt == "":
        #     print(txt_path)

        ff.close()
        if flag is True:
            print("---------------------------")

    pass

r.close()

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
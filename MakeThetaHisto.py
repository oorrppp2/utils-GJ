from scipy import io
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys

# np.set_printoptions(threshold=sys.maxsize)

# fig = plt.figure()
# ax = Axes3D(fig)

classes = [
    "bed",
    "books",
    "book",
    "bottle",
    "bowl",
    "plate",
    "box",
    "cabinet",
    "drawer",
    "kitchen_cabinet",
    "file_cabinet",
    "chair",
    "kitchen_counter",
    "counter",
    "cup",
    "mug",
    "desk",
    "door",
    "lamp",
    "night_stand",
    "painting",
    "picture",
    "pillow",
    "plant",
    "plants",
    "flowers",
    "shelf",
    "bookshelf",
    "sofa_chair",
    "sofa",
    "table",
    "endtable",
    "coffee_table",
    "dining_table",
    "toilet",
    "tv"
]

mat_file = io.loadmat('/home/user/SUNRGBD/SUNRGBDMeta.mat')


for range_index in range(4) :

    data = mat_file.popitem()
    if(data[0] == 'SUNRGBDMeta'):

        sample_size = np.arange(10335)
        # sample_size = np.arange(105)
        histo_path = "/home/user/histo_result.jpg"
        histo_arr = np.zeros(360)
        histo_arr_chair = np.zeros(360)
        histo_arr_bed = np.zeros(360)
        histo_arr_drawer = np.zeros(360)
        histo_arr_sofa = np.zeros(360)
        histo_arr_table = np.zeros(360)
        histo_arr_desk= np.zeros(360)
        histo_arr_lamp = np.zeros(360)
        histo_arr_pillow = np.zeros(360)

        for ele in sample_size:
            image_index = ele
            if (data[1][0][image_index][1].size < 1):
                continue
                pass

            for groundtruth3DBB in data[1][0][image_index][1]:
                for items in groundtruth3DBB:
                    if (items[7].size < 1):
                        continue
                    label = items[3][0]
                    if label in classes:
                        if label == "books":
                            label = "book"
                            pass
                        elif label == "bottle" or label == "bowl" or label == "plate" or label == "cup" or label == "mug":
                            label = "dishware"
                            pass
                        elif label == "cabinet" or label == "kitchen_cabinet" or label == "file_cabinet" or label == "night_stand":
                            label = "drawer"
                            pass
                        elif label == "kitchen_counter":
                            label = "counter"
                            pass
                        elif label == "picture":
                            label = "painting"
                            pass
                        elif label == "plants" or label == "flowers":
                            label = "plant"
                            pass
                        elif label == "bookshelf":
                            label = "shelf"
                            pass
                        elif label == "sofa_chair":
                            label = "sofa"
                            pass
                        elif label == "endtable" or label == "coffee_table" or label == "dining_table":
                            label = "table"
                            pass
                        pass
                    else:
                        continue
                        pass

                    orientation = items[6][0]

                    theta = math.atan2(orientation[1], orientation[0])
                    dtheta = math.degrees(theta)
                    rTheta = np.round(dtheta)
                    # print(type(int(rTheta)))
                    if rTheta == 180:
                        rTheta = 179
                        pass

                    if np.isnan(rTheta):
                        print(items[3][0], ":", dtheta)
                        pass
                    else:
                        index = int(rTheta+180)
                        histo_arr[index] += 1
                        if label == "chair":
                            histo_arr_chair[index] += 1
                            pass
                        if label == "table":
                            histo_arr_table[index] += 1
                            pass
                        if label == "lamp":
                            histo_arr_lamp[index] += 1
                            pass
                        if label == "desk":
                            histo_arr_desk[index] += 1
                            pass
                        if label == "bed":
                            histo_arr_bed[index] += 1
                            pass
                        if label == "drawer":
                            histo_arr_drawer[index] += 1
                            pass
                        if label == "pillow":
                            histo_arr_pillow[index] += 1
                            pass
                        if label == "sofa":
                            histo_arr_sofa[index] += 1
                            pass

                    # print(items[3][0], ":", dtheta)
                    # print("ㄴ", items[3][0], ":", np.round(dtheta))

                    pass
                pass
            pass

        x2 = np.arange(0, 360)
        y2 = histo_arr_table
        x3 = np.arange(0, 360)
        y3 = histo_arr_desk
        x4 = np.arange(0, 360)
        y4 = histo_arr_lamp
        x5 = np.arange(0, 360)
        y5 = histo_arr_pillow
        # plt1, = plt.plot(x1, y1, 'or', label='all')
        # plt2, = plt.plot(x2, y2, 'or', label='chair')
        # plt3, = plt.plot(x3, y3, 'og', label='bed')
        # plt4, = plt.plot(x4, y4, 'ob', label='drawer')
        # plt5, = plt.plot(x5, y5, 'oy', label='sofa')

        plt2, = plt.plot(x2, y2, 'or', label='table')
        plt3, = plt.plot(x3, y3, 'og', label='desk')
        plt4, = plt.plot(x4, y4, 'ob', label='lamp')
        plt5, = plt.plot(x5, y5, 'oy', label='pillow')
        legend_all = plt.legend(handles=[plt2, plt3, plt4, plt5], loc='upper right')
        # art_legend = plt.g
        plt.show()

    else:
        continue
    pass


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

ID = {"bed": 0, "book": 1, "dishware": 2, "box": 3, "drawer": 4, "chair": 5,
      "counter": 6, "desk": 7, "door": 8, "lamp": 9, "painting": 10, "pillow": 11,
      "plant": 12, "shelf": 13, "sofa": 14, "table": 15, "toilet": 16, "tv": 17}
ID_cnt_train = np.zeros(18)
ID_cnt_valid = np.zeros(18)
# print("ID_cnt :", ID_cnt)
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
        index = 0

        pers = 0

        for ele in sample_size:
            # image_index = 0
            # print(ele, "번 째")
            image_index = ele
            if (data[1][0][image_index][1].size < 1):
                continue
                pass
            if image_index % 1000 == 0 and pers < 100:
                pers+=10
                print(pers,"% 진행")


            txt = ""
            check = []
            for groundtruth3DBB in data[1][0][image_index][1]:
                for items in groundtruth3DBB:

                    if (items[7].size < 1):
                        continue

                    """"check"""
                    label = items[3][0]
                    check.append(label)
                    # if label == "coffee_table":
                    #     print("img : ", image_index)

                    """ 3D bounding box """

                    pass

                pass

            if "chair" in check:
                if "sofa" in check:
                    if "table" in check:
                        if "coffee_table" in check:
                            print("***", image_index, "***")
                            print(check)

            # print(check)
            # print("==================")

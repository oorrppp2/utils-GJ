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

                    """ 3D bounding box """

                    # orientation = items[6][0]
                    # # print(orientation)
                    # theta = math.atan2(orientation[1], orientation[0])
                    #
                    # # basis = items[0][inds, :]
                    # # coeffs = items[1][0, inds]
                    #
                    # centroid = items[2]
                    #
                    # t_re = orientation[0]
                    # t_im = orientation[1]
                    #
                    # x = centroid[0,0]
                    # y = centroid[0,1]
                    # z = centroid[0,2]
                    #
                    # """
                    #     txt = ID[label] t_re t_im c1 c2 c3 x y z
                    #     -1 <= t_re, t_im <= 1
                    #     0 <= c1 c2 c3 <= 1
                    #
                    # """
                    #
                    # txt += str(ID[label])
                    # txt += " " + (str(t_re)+"000000")[:8]
                    # txt += " " + (str(t_im)+"000000")[:8]
                    #
                    # txt += " " + (str(c1)+"000000")[:8]
                    # txt += " " + (str(c2)+"000000")[:8]
                    # txt += " " + (str(c3)+"000000")[:8]
                    #
                    # txt += " " + (str(x)+"000000")[:8]
                    # txt += " " + (str(y)+"000000")[:8]
                    # txt += " " + (str(z)+"000000")[:8] + "\n"

                    """ 2D bounding box """
                    gt2DBB = items[7][0]
                    bx = (items[7][0][0] + items[7][0][2] / 2) / col
                    by = (items[7][0][1] + items[7][0][3] / 2) / row
                    bw = items[7][0][2] / col
                    bh = items[7][0][3] / row
                    orientation = items[6][0]
                    t_re = (orientation[0] + 1.0) / 2.0
                    t_im = (orientation[1] + 1.0) / 2.0

                    # print(label, " : ", items[7][0][0], items[7][0][1], items[7][0][2], items[7][0][3])
                    # print(t_re, ", ", t_im)

                    # print(label, " :: ", bx, by, bw, bh)

                    txt += str(ID[label])
                    txt += " " + (str(bx) + "000000")[:8]
                    txt += " " + (str(by) + "000000")[:8]
                    txt += " " + (str(bw) + "000000")[:8]
                    txt += " " + (str(bh) + "000000")[:8]
                    txt += " " + (str(t_re) + "000000")[:8]
                    txt += " " + (str(t_im) + "000000")[:8] + "\n"

                    if max_size < 1901 and random_arr[image_index] == 1:
                        ID_cnt_valid[ID[label]] += 1
                        pass
                    else:
                        ID_cnt_train[ID[label]] += 1
                        pass

                    # print(label, " theta : ", math.atan2(t_im*2-1, t_re*2-1)*180.0/math.pi)
                    # print("--------------------------------")
                    pass

                pass
            print("==================")

            # print(txt)
            # print("t_re : ", txt[-17:-9] , " ||  t_im : ", txt[-8:])
            splitlinestr = str.splitlines(txt)
            for text in splitlinestr:
                tre = text[-17]
                tim = text[-8]
                # if tre != "0":
                #     # print("t_re : ", text[-17:-9])
                #     pass
                # if tim != "0":
                #    # print("t_im : ", text[-8:])
                #     pass

                if tre != "0" or tim != "0":
                    print("img num : ", image_index)
                    print(txt)



                # print("tre type : ", type(tre), " , tre : ", tre)
                # print("t_re : ", text[-17:-9] , " ||  t_im : ", text[-8:])
            # count_num_str = ""
            # for i in range(1143):
            #     # print(splitlinestr[i])
            #     count_num_str += str(splitlinestr[i]) + "\t"
            #     count_num_str += str(count_num[i]) + "\n"
            #
            # print(count_num_str)
            # print("classes size : ",len(classes))


            # if txt != "":
            #     if max_size < 1901 and random_arr[image_index] == 1:
            #         max_size += 1
            #         f = open(txt_s_valid, mode='wt')
            #         f.write(txt)
            #         f.close()
            #
            #         depthInpaint = depth_raw / 65535 * 255
            #         depthInpaint = depthInpaint.astype("uint8")
            #
            #         transformed = np.dstack([color_raw, depthInpaint])
            #         print(img_s_valid)
            #         cv2.imwrite(img_s_valid, transformed)
            #         valid_txt += "data/pose_yolo_valid/" + str(ele) + ".tiff" + "\n"
            #         pass
            #     else:
            #         f = open(txt_s, mode='wt')
            #         f.write(txt)
            #         f.close()
            #
            #         depthInpaint = depth_raw / 65535 * 255
            #         depthInpaint = depthInpaint.astype("uint8")
            #
            #         transformed = np.dstack([color_raw, depthInpaint])
            #         print(img_s)
            #         cv2.imwrite(img_s, transformed)
            #         train_txt += "data/pose_yolo/" + str(ele) + ".tiff" + "\n"
            #         pass
            #     # print(txt)
            #     # print("==================================")
            #     pass
            # else:
            #     continue
            #     pass

            if max_size == 1901:
                print("last index : ", image_index)
                max_size += 1
                pass

            pass

        # print(classes_txt)
        # splitlinestr = str.splitlines(classes_txt)
        # print(index)
        # count_num_str = ""
        # for i in range(1143):
        #     # print(splitlinestr[i])
        #     count_num_str += str(splitlinestr[i]) + "\t"
        #     count_num_str += str(count_num[i]) + "\n"
        #
        # print(count_num_str)
        # print("classes size : ",len(classes))
        #
        # cf = open("/home/user/count.txt", mode='wt')
        # cf.write(count_num_str)
        # cf.close()

        # cf = open(cls_s, mode='wt')
        # cf.write(classes_txt)
        # cf.close()

        print(ID_cnt_train)
        print(ID_cnt_valid)

        # tf = open(train_s, mode='wt')
        # tf.write(train_txt)
        # tf.close()
        #
        # tf = open(valid_s, mode='wt')
        # tf.write(valid_txt)
        # tf.close()

        # print("end")
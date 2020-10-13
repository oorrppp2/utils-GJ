from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys

# np.set_printoptions(threshold=sys.maxsize)


fig = pyplot.figure()
ax = Axes3D(fig)

mat_file = io.loadmat('/home/user/SUNRGBD/SUNRGBDMeta.mat')

for range_index in range(4) :

    data = mat_file.popitem()
    image_index = 0
    for image_index in range(10335):
        if(data[0] == 'SUNRGBDMeta'):
            image_path = data[1][0][image_index][5][0]
            depth_path = data[1][0][image_index][4][0]
            # print("Img path : ", data[1][0][image_index][5][0])

            # color_raw = cv2.imread(data[1][0][image_index][5][0], cv2.COLOR_BGR2RGB)
            # color_raw = cv2.imread(data[1][0][image_index][5][0], cv2.COLOR_RGB2BGR)
            # depth_raw = cv2.imread(data[1][0][image_index][4][0], -1)

            # print(data[1][0][image_index][4][0][-12])

            # if data[1][0][image_index][4][0][-12] != '/':
            #     print(data[1][0][image_index][4][0])
            # data[1][0][image_index][4][0]

            depth_bfx_path = ""
            for i in range(1, len(data[1][0][image_index][4][0])):
                # print(str(-i))
                # print(data[1][0][image_index][4][0][-i])
                if data[1][0][image_index][4][0][-i] is "/":
                    depth_bfx_path = data[1][0][image_index][4][0][:-i]+"_bfx"+ data[1][0][image_index][4][0][-i:]
                    # img_index = i
                    break
            print("depth_bfx_path : " + str(depth_bfx_path))

            # color_raw = cv2.imread(image_path, cv2.COLOR_RGB2BGR)
            # depth_raw = cv2.imread(depth_bfx_path, -1)


            # rgb = np.reshape(color_raw, (len(color_raw)*len(color_raw[0]), 3))
            # rgb = rgb.astype(np.uint8)
            # rgb = rgb.astype("float32")
            # rgb = rgb / 255

            # print(rgb)

            """
                Make 3d point cloud by using depth_raw
            """
            # depthInpaint = (depth_raw>>3) | (depth_raw<<(16-3))
            # depthInpaint = depthInpaint.astype("float32")
            # depthInpaint = depthInpaint / 1000
            # for row in depthInpaint :
            #     for ele in row :
            #         ele = 8 if ele > 8 else ele
            #         pass
            #     pass

            K = data[1][0][image_index][3]
            cx = K[0][2]
            cy = K[1][2]
            fx = K[0][0]
            fy = K[1][1]

            print("image index : " , image_index)
            print("K : " )
            print(K)
        #
        #     range_x = np.arange(0, len(depth_raw[0]))
        #     range_y = np.arange(0, len(depth_raw))
        #
        #     x, y = np.meshgrid(range_x, range_y)
        #
        #     x3 = (x-cx)*depthInpaint*1/fx
        #     y3 = (y-cy)*depthInpaint*1/fy
        #     z3 = depthInpaint
        #
        #     x3 = np.reshape(x3, len(x3)*len(x3[0]))
        #     y3 = np.reshape(y3, len(y3)*len(y3[0]))
        #     z3 = np.reshape(z3, len(z3)*len(z3[0]))
        #
        #     pointsMat = np.vstack((x3,z3,-y3))
        #
        #     # remove nan
        #     nan_index = []
        #     for i in range(len(x3)):
        #         # if x3[i] != 0 or y3[i] != 0 or z3[i] != 0:
        #         if x3[i] == 0 and y3[i] == 0 and z3[i] == 0:
        #             nan_index.append(i)
        #             # nan_flag[i] = True
        #             # print(pointsMat[:,i])
        #             pass
        #         pass
        #     pointsMat = np.delete(pointsMat, nan_index, axis=1)
        #     rgb = np.delete(rgb, nan_index, axis=0)
        #
        #     Rtilt = data[1][0][image_index][2]
        #     point3d = Rtilt @ pointsMat
        #
        #     x3 = point3d[0]
        #     y3 = -point3d[2]
        #     z3 = point3d[1]
        #
        #     x2 = (fx*x3 + cx*z3) / z3
        #     y2 = (fy*y3 + cy*z3) / z3
        #
        #     xmin, xmax, ymin, ymax = 10000, 0, 10000, 0
        #     for i in range(len(x2)):
        #         if x2[i] < xmin:
        #             xmin = x2[i]
        #         if x2[i] > xmax:
        #             xmax = x2[i]
        #         if y2[i] < ymin:
        #             ymin = y2[i]
        #         if y2[i] > ymax:
        #             ymax = y2[i]
        #
        #     # print(xmin, xmax, ymin, ymax)
        #
        #     x2 -= xmin
        #     xmax -= xmin
        #     y2 -= ymin
        #     ymax -= ymin
        #
        #     x2 = x2.astype(int)
        #     y2 = y2.astype(int)
        #
        #     img = np.zeros((int(ymax)+1, int(xmax)+1, 3), np.uint8)
        #     # for i in range(int(xmax)+1):
        #     #     for j in range(int(ymax)+1):
        #     #         img.itemset(j,i,0,255)
        #     #         img.itemset(j,i,1,255)
        #     #         img.itemset(j,i,2,255)
        #
        #     for i in range(0, len(x2)):
        #         img.itemset(y2[i], x2[i], 0, rgb[i, 0])
        #         img.itemset(y2[i], x2[i], 1, rgb[i, 1])
        #         img.itemset(y2[i], x2[i], 2, rgb[i, 2])
        #
        #     cv2.imshow("image", img)
        #     cv2.waitKey(0)
        #     # cv2.imwrite("/home/user/PycharmProjects/imgs/untilted_img/"+str(image_index)+".jpg", img)
        #
        #     # for i in range(1,int(xmax)):
        #     #     for j in range(1,int(ymax)):
        #     #         if img[j,i,0] + img[j,i,1] + img[j,i,2] == 0:
        #     #             div_num = 0
        #     #             r=0
        #     #             g=0
        #     #             b=0
        #     #             if img[j-1,i,0] + img[j-1,i,1] + img[j-1,i,2] != 0:
        #     #                 div_num += 1
        #     #                 r += img[j-1,i,0]
        #     #                 g += img[j-1,i,1]
        #     #                 b += img[j-1,i,2]
        #     #             if img[j-1,i-1,0] + img[j-1,i-1,1] + img[j-1,i-1,2] != 0:
        #     #                 div_num += 1
        #     #                 r += img[j-1,i-1,0]
        #     #                 g += img[j-1,i-1,1]
        #     #                 b += img[j-1,i-1,2]
        #     #             if img[j-1,i+1,0] + img[j-1,i+1,1] + img[j-1,i+1,2] != 0:
        #     #                 div_num += 1
        #     #                 r += img[j-1,i+1,0]
        #     #                 g += img[j-1,i+1,1]
        #     #                 b += img[j-1,i+1,2]
        #     #
        #     #             if img[j+1,i,0] + img[j+1,i,1] + img[j+1,i,2] != 0:
        #     #                 div_num += 1
        #     #                 r += img[j+1,i,0]
        #     #                 g += img[j+1,i,1]
        #     #                 b += img[j+1,i,2]
        #     #             if img[j+1,i-1,0] + img[j+1,i-1,1] + img[j+1,i-1,2] != 0:
        #     #                 div_num += 1
        #     #                 r += img[j+1,i-1,0]
        #     #                 g += img[j+1,i-1,1]
        #     #                 b += img[j+1,i-1,2]
        #     #             if img[j+1,i+1,0] + img[j+1,i+1,1] + img[j+1,i+1,2] != 0:
        #     #                 div_num += 1
        #     #                 r += img[j+1,i+1,0]
        #     #                 g += img[j+1,i+1,1]
        #     #                 b += img[j+1,i+1,2]
        #     #
        #     #
        #     #             if img[j,i-1,0] + img[j,i-1,1] + img[j,i-1,2] != 0:
        #     #                 div_num += 1
        #     #                 r += img[j,i-1,0]
        #     #                 g += img[j,i-1,1]
        #     #                 b += img[j,i-1,2]
        #     #             if img[j,i+1,0] + img[j,i+1,1] + img[j,i+1,2] != 0:
        #     #                 div_num += 1
        #     #                 r += img[j,i+1,0]
        #     #                 g += img[j,i+1,1]
        #     #                 b += img[j,i+1,2]
        #     #
        #     #             if div_num == 0:
        #     #                 continue
        #     #
        #     #             r /= div_num
        #     #             g /= div_num
        #     #             b /= div_num
        #     #
        #     #             img.itemset(j,i,0,int(r))
        #     #             img.itemset(j,i,1,int(g))
        #     #             img.itemset(j,i,2,int(b))
        #
        #     # cv2.imwrite("/home/user/PycharmProjects/imgs/untilted_img_interpolated/"+str(image_index)+".jpg", img)
        #
        #     # cv2.imshow("image_interpolation", img)
        #     # cv2.waitKey(0)
        #     # cv2.imshow("ori image", color_raw)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #
        else:
            continue
        # pass
        #

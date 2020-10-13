from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys
import os
from utils.image import get_affine_transform
from utils.ddd_utils import compute_box_3d, project_to_image, project_to_image_sun2, project_to_image_sun, alpha2rot_y, compute_box_3d_sun, compute_box_3d_sun_2, compute_box_3d_sun_3,compute_box_3d_sun_4,compute_box_3d_sun_5,compute_box_3d_sun_6
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d, draw_box_3d_sun
from utils.ddd_utils import rot_y2alpha
from numpy.linalg import inv

# np.set_printoptions(threshold=sys.maxsize)
def calculateDistance(x1, y1, x2, y2):
    dist =math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist
def bound_check(x_boundary, y_boundary, width, height):
    if (x_boundary < 0 or x_boundary >= width  or y_boundary < 0 or y_boundary >= height):
        return 0
    return 1
def draw_2d_box(color_raw, xmin, ymin, xmax, ymax, c=(0, 255, 0)):
    cv2.line(color_raw, (int(xmin), int(ymin)), (int(xmax), int(ymin)), c, 2, lineType=cv2.LINE_AA)
    cv2.line(color_raw, (int(xmax), int(ymin)), (int(xmax), int(ymax)), c, 2, lineType=cv2.LINE_AA)
    cv2.line(color_raw, (int(xmax), int(ymax)), (int(xmin), int(ymax)), c, 2, lineType=cv2.LINE_AA)
    cv2.line(color_raw, (int(xmin), int(ymax)), (int(xmin), int(ymin)), c, 2, lineType=cv2.LINE_AA)
    ct_x = (xmin + xmax) / 2
    ct_y = (ymin + ymax) / 2
    cv2.circle(color_raw, (int(ct_x), int(ct_y)), 3, c, 2, lineType=cv2.LINE_AA)

def compute_3d_box(color_raw, dim, centroid, rot_mat, Rtilt):
    x = dim[0] / 2
    y = dim[1] / 2
    z = dim[2] / 2
    x_corners = [-x, x, x, -x, -x, x, x, -x]
    y_corners = [y, y, -y, -y, y, y, -y, -y]
    z_corners = [z, z, z, z, -z, -z, -z, -z]
    corners2 = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(rot_mat, corners2)  # (3,8)
    corners_3d = corners_3d + np.array(centroid, dtype=np.float32).reshape(3, 1)  # (3,8)
    corners_3d = corners_3d.transpose(1, 0)
    sun_permutation = [0, 2, 1]
    index = np.argsort(sun_permutation)
    corners_3d = corners_3d[:, index]
    corners_3d[:, 1] *= -1
    # corners_3d = np.array(corners_3d, dtype=np.int64)
    return corners_3d

def compute_3d_box_untilted(dim, centroid, rot_mat):
    x = dim[0] / 2
    y = dim[1] / 2
    z = dim[2] / 2
    x_corners = [-x, x, x, -x, -x, x, x, -x]
    y_corners = [y, y, -y, -y, y, y, -y, -y]
    z_corners = [z, z, z, z, -z, -z, -z, -z]
    corners2 = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(rot_mat, corners2)  # (3,8)

    # corners_3d = np.dot(np.transpose(Rtilt), corners_3d)

    # camera_coordinate
    corners_3d = corners_3d.transpose(1, 0)
    sun_permutation = [0, 2, 1]
    index = np.argsort(sun_permutation)
    corners_3d = corners_3d[:, index]
    corners_3d[:, 1] *= -1

    corners_3d = np.transpose(corners_3d) + np.array(centroid, dtype=np.float32).reshape(3, 1)  # (3,8)

    # corners_3d = np.array(corners_3d, dtype=np.int64)
    return corners_3d


def draw_box_3d(image, corners, c=(0, 0, 255)):
  #kitti용
  # face_idx = [[0,1,5,4], #앞
  #             [1,2,6,5], #왼
  #             [2,3,7,6],  #뒤
  #             [3,0,4,7]] #오
  #sun rgbd용
  face_idx = [[1,2,6,5], #앞
              [6,5,4,7], #왼
              [7,4,0,3],  #뒤
              [3,0,1,2]] #오
  # face_idx = [[[2,3,7,6]],
  #              [3,0,4,7],
  #              [[0,1,5,4]],
  #              [1,2,6,5]]
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      cv2.line(image, ((corners[f[j], 0]), (corners[f[j], 1])),
               ((corners[f[(j+1)%4], 0]), (corners[f[(j+1)%4], 1])), c, 2, lineType=cv2.LINE_AA)


def project_3d_bbox(box_2d):
    xmin_projected = int(box_2d.transpose()[0].min())
    xmax_projected = int(box_2d.transpose()[0].max())
    ymin_projected = int(box_2d.transpose()[1].min())
    ymax_projected = int(box_2d.transpose()[1].max())

    return xmin_projected, xmax_projected, ymin_projected, ymax_projected
fig = pyplot.figure()
ax = Axes3D(fig)

mat_file = io.loadmat('C:/SUNRGBDMeta/SUNRGBDMeta.mat')
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
class_final = [
    "bathtub",
    "bed",
    "bookshelf",
    "chair",
    "desk",
    "dresser",
    "nightstand",
    "sofa",
    "table",
    "toilet",
]
for range_index in range(4) :
    data = mat_file.popitem()
    if(data[0] == 'SUNRGBDMeta'):
        for i in range(1201, 10335):
            image_index = i
            #원하는 iindex
            # image_index = 3
            if i in skip_img:
                continue
            print("Img path : ", data[1][0][image_index][5][0])
            path = data[1][0][image_index][5][0]
            if '//' in path:
                path = path.replace('//', '/')
            path_split = path.split('/')
            out_str = ''
            for i in range(0, len(path_split)-2):
                out_str = out_str + path_split[i]
                out_str += '/'
            out_str = out_str + 'depth_bfx/'
            print(out_str)
            file_list = os.listdir(out_str)
            print(file_list)
            out_str = out_str + file_list[-1]
            # print("revised path", path)
            # path = path.split('/')
            # print(path)
            # out_str = ''
            # for i in range(0, len(path)-2):
            #     out_str = out_str + path[i]
            #     out_str += '/'
            #
            # file_name =  path[-1].split('.')[0]
            # print(file_name)
            # print(out_str)
            # if 'kv1/b3dodata' in data[1][0][image_index][5][0]:
            #     file_name = file_name + '_abs'
            # out_str = out_str + 'depth_bfx/' + file_name + '.png'


            # color_raw = cv2.imread(data[1][0][image_index][5][0], cv2.COLOR_BGR2RGB)
            color_raw = cv2.imread(data[1][0][image_index][5][0], cv2.COLOR_RGB2BGR)
            # depth_raw_ori = cv2.imread(data[1][0][image_inde ][4][0], -1)
            depth_raw = cv2.imread(out_str, -1)
            height, width, channel = color_raw.shape
            #color_raw : (height, width, channel) (531, 681, 3)
            rgb = np.reshape(color_raw, (len(color_raw)*len(color_raw[0]), 3))
            rgb2 = np.reshape(color_raw, ((width * height), 3))
            rgb = rgb.astype(np.uint8)
            # rgb = rgb.astype("float32")
            # rgb = rgb / 255

            # print(rgb)

            """
                Make 3d point cloud by using depth_raw
            """
            #끝에 값을 잘라내는것
            depthInpaint = (depth_raw>>3) | (depth_raw<<(16-3))
            depthInpaint = depthInpaint.astype("float32")
            depthInpaint = depthInpaint / 1000
            # for row in depthInpaint :
            #     for ele in row :
            #         ele = 8 if ele > 8 else ele
            #         pass
            #     pass

            K = np.array(data[1][0][i][3])
            new_K = np.zeros((K.shape[0], K.shape[1] + 1))
            new_K[:, :-1] = K  #(3,3)은 원래데이터            cx = K[0][2]
            cx = K[0][2]
            cy = K[1][2]
            fx = K[0][0]
            fy = K[1][1]
            # print(K)
            range_x = np.arange(0, len(depth_raw[0]))
            range_x2 = np.arange(0, width )
            range_y = np.arange(0, len(depth_raw))
            range_y2 = np.arange(0, height )
            x, y = np.meshgrid(range_x, range_y)
            z3 = depthInpaint

            x3 = (x-cx)*depthInpaint*1/fx
            y3 = (y-cy)*depthInpaint*1/fy

            x3 = np.reshape(x3, len(x3)*len(x3[0]))
            y3 = np.reshape(y3, len(y3)*len(y3[0]))
            z3 = np.reshape(z3, len(z3)*len(z3[0]))

            pointsMat = np.vstack((x3,z3,-y3))

            # remove nan
            nan_index = []
            for i in range(len(x3)):
                # if x3[i] != 0 or y3[i] != 0 or z3[i] != 0:
                if x3[i] == 0 and y3[i] == 0 and z3[i] == 0:
                    nan_index.append(i)
                    # nan_flag[i] = True
                    # print(pointsMat[:,i])
                    pass
                pass

            pointsMat = np.delete(pointsMat, nan_index, axis=1)
            rgb = np.delete(rgb, nan_index, axis=0)

            Rtilt = data[1][0][image_index][2]
            point3d = Rtilt @ pointsMat
            # point3d = pointsMat
            x3 = point3d[0]
            y3 = -point3d[2]
            z3 = point3d[1]

            x2 = fx*x3 + cx*z3
            y2 = fy*y3 + cy*z3
            x2 = x2 / z3
            y2 = y2 / z3

            i_xmin, i_xmax, i_ymin, i_ymax = 10000, 0, 10000, 0
            for i in range(len(x2)):
                if x2[i] < i_xmin:
                    i_xmin = x2[i]
                if x2[i] > i_xmax:
                    i_xmax = x2[i]
                if y2[i] < i_ymin:
                    i_ymin = y2[i]
                if y2[i] > i_ymax:
                    i_ymax = y2[i]
            # print(i_xmin, i_xmax, i_ymin, i_ymax)

            x_under = np.floor(x2).astype(int)
            x_upper = x_under + 1
            y_under = np.floor(y2).astype(int)
            y_upper = y_under + 1


            x2 = x2.astype(int)
            y2 = y2.astype(int)


            # print("x2_minus")
            #x2 & y2 val : pixel position ex. wxh = 730 x 530 = 386900
            #print(x2_minus, y2_minus
            # x2_minus = [(x2[i], y2[i]) for i in range(len(x2)) if x2[i] < 0]
            # print(x2_minus)
            # print("y2_minus")
            # print(len(x2), print(len(y2)))
            # print(y2[i] for i in range(len(y2)) if y2[i] < 0)
            img = np.zeros((int(i_ymax)+1, int(i_xmax)+1, 3), np.uint8)
            # img = np.zeros((height, width, 3), np.uint8)
            # img2 = np.zeros((height, width, 3), np.uint8)
            # img3 = np.zeros((height, width, 3), np.uint8)
            img2 = np.zeros((int(i_ymax)+1, int(i_xmax)+1, 3), np.uint8)
            img3 = np.zeros((int(i_ymax)+1, int(i_xmax)+1, 3), np.uint8)
            img4 = np.zeros((int(i_ymax)+1, int(i_xmax)+1, 3), np.uint8)

            # print(img.shape)
            img_height, img_width, img_channel = img.shape
            size_compare_ratio = [img_width/width, img_height/height]
            # print((size_compare_ratio)
            #forward warping
            for i in range(0, len(x2)):
                if(x2[i] < 0 or y2[i]<0):
                    continue
                img.itemset(y2[i], x2[i], 0, rgb[i, 0])
                img.itemset(y2[i], x2[i], 1, rgb[i, 1])
                img.itemset(y2[i], x2[i], 2, rgb[i, 2])
            # cv2.imshow("img",img)
            # cv2.waitKey()

            # dst=cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 21, 35)
            # dst1 = cv2.medianBlur(img, 3)
            # # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            # dst_k = cv2.filter2D(dst1, -1, kernel)
            # dst2 = cv2.GaussianBlur(img, (3,3), 0)
            # dst3 = cv2.bilateralFilter(img, 9, 75, 75)
            # cv2.imshow("dst1", dst1)
            # cv2.imshow("dstk", dst_k)
            # cv2.imshow("dst2", dst2)
            # cv2.imshow("dst3", dst3)
            # cv2.waitKey()
            # ct_src = np.array([img.shape[1] / 2., img.shape[0] / 2.])  # width, height순으로 이미지의 center point를 구함.
            # s = np.array([width, height], dtype=np.int32)  # (448, 448)
            # trans_input = get_affine_transform(ct_src, s, 0, [width, height])
            # inp = cv2.warpAffine(img, trans_input,
            #                      (width, height),
            #                      flags=cv2.INTER_LINEAR)

            # cv2.imshow("augmentation", inp)
            # cv2.waitKey()

            # for i in range(int(xmax)+1):
            #     for j in range(int(ymax)+1):
            #         img.itemset(j,i,0,255)
            #         img.itemset(j,i,1,255)
            #         img.itemset(j,i,2,255)

            label_lists = []

            for groundtruth3DBB in data[1][0][image_index][1]:
                img_flag =False
                for items in groundtruth3DBB:
                    bbox2DFlag = items[7].size
                    if (bbox2DFlag >= 1):
                        label = items[3][0]
                        if label not in class_final:
                            continue

                        #ground_truth_2d_bbox(green)
                        bbox2D = items[7][0]
                        xmin = bbox2D[0]
                        ymin = bbox2D[1]
                        xmax = int(bbox2D[0]) + int(bbox2D[2]) if int(bbox2D[0]) + int(bbox2D[2]) < width - 1 else width - 1
                        ymax = int(bbox2D[1]) + int(bbox2D[3]) if int(bbox2D[1]) + int(bbox2D[3]) < height - 1 else height - 1
                        width_2d = xmax - xmin
                        height_2d = ymax - ymin
                        print("2d bbox width, height")
                        print(width_2d, height_2d)
                        #consider ratio
                        # width_2d = width_2d * size_compare_ratio[0]
                        # height_2d = height_2d * size_compare_ratio[1]

                        # print('2D box')
                        # print(xmax, ymax)
                        ct_x = xmin + bbox2D[2] / 2
                        ct_y = ymin + bbox2D[3] / 2
                        ct_x_int = int(ct_x)
                        ct_y_int = int(ct_y)
                        ct_xy_coord = int((width * ct_y_int) + ct_x_int)
                        ct_xy_coord_dst = [x2[ct_xy_coord], y2[ct_xy_coord]]
                        # bbox2D_coord_dst_ct_x = (bbox2D[0][0] + bbox2D[3][0]) / 2
                        # bbox2D_coord_dst_ct_y = (bbox2D[0][1] + bbox2D[3][1]) / 2
                        cv2.circle(img, (int(ct_xy_coord_dst[0]), int(ct_xy_coord_dst[1])), 5, (255, 0, 0), -1)
                        draw_2d_box(img, ct_xy_coord_dst[0]- width_2d/2 , ct_xy_coord_dst[1] - height_2d/2,
                                    ct_xy_coord_dst[0] + width_2d/2, ct_xy_coord_dst[1] + height_2d/2 , c=(0, 255, 0))

                        #draw ground truth bbox2D(green)
                        # draw_2d_box(img, xmin, ymin, xmax, ymax)
                        '''
                        # backwarping
                        '''

                        #bbox 4점의 좌표 변환 값
                        bbox2D_src = [int((ymin*width)+xmin), int((ymin*width)+xmax), int((ymax*width)+xmin), int((ymax*width)+xmax)]

                        bbox2D_dst = [[x2[i], y2[i]] for i in bbox2D_src]


                        # image 변환값 (image의 꼭지점 4개)
                        bbox2D_coord_index = [0,
                                              (width-1),
                                              (height -1) *width,
                                              (height-1)*width + (width-1)
                                              ]
                        ind_coord = [[0, 0], #(x_min, y_max)
                                     [width-1, 0], #(x_min, y_min)
                                     [0 , height - 1], #(x_max, ymin)
                                     [width -1 , height -1]
                                     ]
                        bbox2D_coord_dst = [[x2[i], y2[i]] for i in bbox2D_coord_index]
                        # print(bbox2D_coord_index)
                        # cetnerPoint를 기준으로 시계방향
                        # c_int = [int(width/2), int(height/2)]
                        # ind_1 = c_int[1]*width + c_int[0]
                        # ind_2 = (c_int[1]-20)*(width) + c_int[0]
                        # ind_3 =  ind_2 + 20
                        # ind_4 =  ind_1 + 20
                        # bbox2D_coord_index = [ind_1, ind_2, ind_3, ind_4]
                        #
                        # ind_coord = [[c_int[0], c_int[1]], #(x_min, y_max)
                        #              [c_int[0], c_int[1]- 20], #(x_min, y_min)
                        #              [c_int[0] + 20, c_int[1] - 20], #(x_max, ymin)
                        #              [c_int[0] + 20, c_int[1]]
                        #             ]



                        ii = int(height/2 * width/2 + width/2)
                        ii_list = [ii]


                        # for i in range(ii, len(x2), int(width/3)):
                        #     if count < 4:
                        #         if x2[i] > 0 and y2[i] > 0 :
                        #             print(i)
                        #             src_height= int(i/height)
                        #             src_width = int(i%height)
                        #             src_list.append([src_width, src_height])
                        #             dst_list.append([x2[i], y2[i]])
                        #             count += 1
                        #     elif count == 4:
                        #         continue

                        # if count !=4:
                        #     print("count not enough")

                        # bbox2D_coord_ct = [(height-1) * (width-1)]
                        #2D bbox용
                        # bbox2D_coord_src = [[xmin, ymin],
                        #                 [xmax, ymin],
                        #                 [xmin, ymax],
                        #                 [xmax, ymax]]


                        # 임의의 4점 용
                        # bbox2D_coord_src= ind_coord



                        # for i in bbox2D_coord_index:
                        #                         #     if(x2[i]>0 and y2[i]>0):
                        #                         #         bbox2D_coord_dst.append([x2[i],y2[i]])
                        #                         # if len(bbox2D_coord_dst[0]) != 4:
                        #                         #     print("minus in bbox2D_coord_dst")
                        #                         #     print(bbox2D_coord_dst)
                        #                         #     continue
                        # bbox2D_coord_dst = dst_list

                        # print(bbox2D_coord_dst)
                        # if img_flag == False:
                        #     projection_matrix = np.zeros((8, 8))  # 나중에 inverse취한후
                        #     # coord_matrix = np.zeros((8,1)) #얘랑 곱해줘야함
                        #     coord_matrix = np.array([bbox2D_coord_dst[0][0], bbox2D_coord_dst[0][1],
                        #                              bbox2D_coord_dst[1][0], bbox2D_coord_dst[1][1],
                        #                              bbox2D_coord_dst[2][0], bbox2D_coord_dst[2][1],
                        #                              bbox2D_coord_dst[3][0], bbox2D_coord_dst[3][1]
                        #                              ])
                        #
                        #     projection_matrix[0][0] = bbox2D_coord_src[0][0]  # x1
                        #     projection_matrix[0][1] = bbox2D_coord_src[0][1]  # y1
                        #     projection_matrix[0][2] = 1.0
                        #     projection_matrix[0][6] = -1 * bbox2D_coord_dst[0][0] * bbox2D_coord_src[0][0]  # -x1'x1
                        #     projection_matrix[0][7] = -1 * bbox2D_coord_dst[0][0] * bbox2D_coord_src[0][1]  # -x1'y1
                        #
                        #     projection_matrix[1][3] = bbox2D_coord_src[0][0]  # x1
                        #     projection_matrix[1][4] = bbox2D_coord_src[0][1]  # y1
                        #     projection_matrix[1][5] = 1.0
                        #     projection_matrix[1][6] = -1 * bbox2D_coord_dst[0][1] * bbox2D_coord_src[0][0]  # -y1'x1
                        #     projection_matrix[1][7] = -1 * bbox2D_coord_dst[0][1] * bbox2D_coord_src[0][1]  # -y1'y1
                        #
                        #     projection_matrix[2][0] = bbox2D_coord_src[1][0]  # x2
                        #     projection_matrix[2][1] = bbox2D_coord_src[1][1]  # y2
                        #     projection_matrix[2][2] = 1.0
                        #     projection_matrix[2][6] = -1 * bbox2D_coord_dst[1][0] * bbox2D_coord_src[1][0]  # -x2'x2
                        #     projection_matrix[2][7] = -1 * bbox2D_coord_dst[1][0] * bbox2D_coord_src[1][1]  # -x2'y2
                        #
                        #     projection_matrix[3][3] = bbox2D_coord_src[1][0]
                        #     projection_matrix[3][4] = bbox2D_coord_src[1][1]
                        #     projection_matrix[3][5] = 1.0
                        #     projection_matrix[3][6] = -1 * bbox2D_coord_dst[1][1] * bbox2D_coord_src[1][0]
                        #     projection_matrix[3][7] = -1 * bbox2D_coord_dst[1][1] * bbox2D_coord_src[1][1]
                        #
                        #     projection_matrix[4][0] = bbox2D_coord_src[2][0]
                        #     projection_matrix[4][1] = bbox2D_coord_src[2][1]
                        #     projection_matrix[4][2] = 1.0
                        #     projection_matrix[4][6] = -1 * bbox2D_coord_src[2][0] * bbox2D_coord_dst[2][0]
                        #     projection_matrix[4][7] = -1 * bbox2D_coord_src[2][1] * bbox2D_coord_dst[2][0]
                        #
                        #     projection_matrix[5][3] = bbox2D_coord_src[2][0]
                        #     projection_matrix[5][4] = bbox2D_coord_src[2][1]
                        #     projection_matrix[5][5] = 1.0
                        #     projection_matrix[5][6] = -1 * bbox2D_coord_src[2][0] * bbox2D_coord_dst[2][1]
                        #     projection_matrix[5][7] = -1 * bbox2D_coord_src[2][1] * bbox2D_coord_dst[2][1]
                        #
                        #     projection_matrix[6][0] = bbox2D_coord_src[3][0]
                        #     projection_matrix[6][1] = bbox2D_coord_src[3][1]
                        #     projection_matrix[6][2] = 1.0
                        #     projection_matrix[6][6] = -1 * bbox2D_coord_src[3][0] * bbox2D_coord_dst[3][0]
                        #     projection_matrix[6][7] = -1 * bbox2D_coord_src[3][1] * bbox2D_coord_dst[3][0]
                        #
                        #     projection_matrix[7][3] = bbox2D_coord_src[3][0]
                        #     projection_matrix[7][4] = bbox2D_coord_src[3][1]
                        #     projection_matrix[7][5] = 1.0
                        #     projection_matrix[7][6] = -1 * bbox2D_coord_src[3][0] * bbox2D_coord_dst[3][1]
                        #     projection_matrix[7][7] = -1 * bbox2D_coord_src[3][1] * bbox2D_coord_dst[3][1]
                        #
                        #     inv_projection_matrix = inv(projection_matrix)
                        #     perspective_matrix = np.dot(inv_projection_matrix, coord_matrix)
                        #     perspective_matrix = perspective_matrix.tolist()
                        #     perspective_matrix_3d = np.ones((3, 3))
                        #     perspective_matrix_3d[0][0] = perspective_matrix[0]
                        #     perspective_matrix_3d[0][1] = perspective_matrix[1]
                        #     perspective_matrix_3d[0][2] = perspective_matrix[2]
                        #     perspective_matrix_3d[1][0] = perspective_matrix[3]
                        #     perspective_matrix_3d[1][1] = perspective_matrix[4]
                        #     perspective_matrix_3d[1][2] = perspective_matrix[5]
                        #     perspective_matrix_3d[2][0] = perspective_matrix[6]
                        #     perspective_matrix_3d[2][1] = perspective_matrix[7]
                        #
                        #     inv_perspective_matrix_3d = inv(perspective_matrix_3d)

                        # for h in range(0, height):
                        #     for w in range(0, width):
                        #         w_coord = perspective_matrix_3d[2][0] * w + perspective_matrix_3d[2][1] * h + 1
                        #         #only backwarping
                        #         # w_coord = 1
                        #         x_coord = int(
                        #             (perspective_matrix_3d[0][0] * w + perspective_matrix_3d[0][1] * h +
                        #              perspective_matrix_3d[0][2]) / w_coord)
                        #         y_coord = int(
                        #             (perspective_matrix_3d[1][0] * w + perspective_matrix_3d[1][1] * h +
                        #              perspective_matrix_3d[1][2]) / w_coord)
                        #         # x_coord = int(x_coord *size_compare_ratio[0])
                        #         # y_coord = int(y_coord * size_compare_ratio[1])
                        #
                        #         if not bound_check(x_coord, y_coord, img_width, img_height):
                        #             continue
                        #
                        #         img4.itemset(y_coord, x_coord, 0, color_raw[h][w][0])
                        #         img4.itemset(y_coord, x_coord, 1, color_raw[h][w][1])
                        #         img4.itemset(y_coord, x_coord, 2, color_raw[h][w][2])
                        # # # img2에 bwarping
                        # # # img_height, img_width는 dst의 크기기
                        # for h in range(0, img_height):
                        #     for w in range(0, img_width):
                        #         # h = int(h * size_compare_ratio[0])
                        #         # w = int(w * size_compare_ratio[])
                        #         w_coord = inv_perspective_matrix_3d[2][0] * w + inv_perspective_matrix_3d[2][1] * h + inv_perspective_matrix_3d[2][2]
                        #         #only backwarping
                        #         # w_coord = 1
                        #         x_coord = int(
                        #             (inv_perspective_matrix_3d[0][0] * w + inv_perspective_matrix_3d[0][1] * h +
                        #              inv_perspective_matrix_3d[0][2]) / w_coord)
                        #         y_coord = int(
                        #             (inv_perspective_matrix_3d[1][0] * w + inv_perspective_matrix_3d[1][1] * h +
                        #              inv_perspective_matrix_3d[1][2]) / w_coord)
                        #         # x_coord = int(x_coord *size_compare_ratio[0])
                        #         # y_coord = int(y_coord * size_compare_ratio[1])
                        #
                        #         if not bound_check(x_coord, y_coord, width, height):
                        #             continue
                        #
                        #         img2.itemset(h, w, 0, color_raw[y_coord][x_coord][0])
                        #         img2.itemset(h, w, 1, color_raw[y_coord][x_coord][1])
                        #         img2.itemset(h, w, 2, color_raw[y_coord][x_coord][2])
                        #
                        # for h in range(0, img_height):
                        #     for w in range(0, img_width):
                        #         w_coord = inv_perspective_matrix_3d[2][0] * w + inv_perspective_matrix_3d[2][1] * h + inv_perspective_matrix_3d[2][2]
                        #         #only backwarping
                        #         x_coord = (inv_perspective_matrix_3d[0][0] * w + inv_perspective_matrix_3d[0][1] * h +
                        #              inv_perspective_matrix_3d[0][2]) / w_coord
                        #         y_coord = (inv_perspective_matrix_3d[1][0] * w + inv_perspective_matrix_3d[1][1] * h +
                        #              inv_perspective_matrix_3d[1][2]) / w_coord
                        #
                        #         wx_1 = x_coord - math.floor(x_coord)
                        #         wx_0 = 1.0 - wx_1
                        #
                        #         wy_1 = y_coord - math.floor(y_coord)
                        #         wy_0 = 1.0 - wy_1
                        #
                        #         x_coord_int = int(math.floor(x_coord))
                        #         y_coord_int = int(math.floor(y_coord))
                        #         pixel_b, pixel_g, pixel_r, ratio = 0, 0, 0, 0
                        #
                        #         if bound_check(x_coord_int, y_coord_int, width, height):
                        #             pixel_b += wx_0 * wy_0 * color_raw[y_coord_int][x_coord_int][0]
                        #             pixel_g += wx_0 * wy_0 * color_raw[y_coord_int][x_coord_int][1]
                        #             pixel_r += wx_0 * wy_0 * color_raw[y_coord_int][x_coord_int][2]
                        #             ratio += wx_0 * wy_0
                        #
                        #         if bound_check(x_coord_int + 1, y_coord_int, width, height):
                        #             pixel_b += wx_1 * wy_0 * color_raw[y_coord_int][x_coord_int+1][0]
                        #             pixel_g += wx_1 * wy_0 * color_raw[y_coord_int][x_coord_int+1][1]
                        #             pixel_r += wx_1 * wy_0 * color_raw[y_coord_int][x_coord_int+1][2]
                        #             ratio += wx_1 * wy_0
                        #
                        #         if bound_check(x_coord_int, y_coord_int + 1, width, height):
                        #             pixel_b += wx_0 * wy_1 * color_raw[y_coord_int+1][x_coord_int][0]
                        #             pixel_g += wx_0 * wy_1 * color_raw[y_coord_int+1][x_coord_int][1]
                        #             pixel_r += wx_0 * wy_1 * color_raw[y_coord_int+1][x_coord_int][2]
                        #             ratio += wx_0 * wy_1
                        #
                        #         if bound_check(x_coord_int + 1, y_coord_int + 1, width, height):
                        #             pixel_b += wx_1 * wy_1 * color_raw[y_coord_int+1][x_coord_int+1][0]
                        #             pixel_g += wx_1 * wy_1 * color_raw[y_coord_int+1][x_coord_int+1][1]
                        #             pixel_r += wx_1 * wy_1 * color_raw[y_coord_int+1][x_coord_int+1][2]
                        #             ratio += wx_1 * wy_1
                        #         if(ratio == 0):
                        #             ratio = 0.000000001
                        #         # print(ratio)
                        #         pixel_b = math.floor(pixel_b/ratio + 0.5)
                        #         pixel_g = math.floor(pixel_g / ratio + 0.5)
                        #         pixel_r = math.floor(pixel_r / ratio + 0.5)
                        #
                        #         img3.itemset(h, w, 0, pixel_b)
                        #         img3.itemset(h, w, 1, pixel_g)
                        #         img3.itemset(h, w, 2, pixel_r)
                        #
                        # img_flag = True
                        #############
                        # bbox2D = np.dot(inv_perspective_matrix_3d, np.transpose(bbox2D_coord))
                        # bbox2D = bbox2D.tolist()
                        # print(bbox2D)
                        ###############
                        #rotated 2d bbox
                        top_width = calculateDistance(int(bbox2D_dst[0][0]), int(bbox2D_dst[0][1]), int(bbox2D_dst[1][0]), int(bbox2D_dst[1][1]))
                        right_height = calculateDistance(int(bbox2D_dst[1][0]), int(bbox2D_dst[1][1]), int(bbox2D_dst[3][0]), int(bbox2D_dst[3][1]))
                        bottom_width = calculateDistance(int(bbox2D_dst[2][0]), int(bbox2D_dst[2][1]), int(bbox2D_dst[3][0]), int(bbox2D_dst[3][1]))
                        left_height = calculateDistance(int(bbox2D_dst[0][0]), int(bbox2D_dst[0][1]), int(bbox2D_dst[2][0]), int(bbox2D_dst[2][1]))
                        bbox3DFlag = True
                        if abs(top_width - bottom_width) > width_2d or abs(left_height - right_height) > height_2d :
                            bbox3DFlag = False

                        untilt_width_mean = (top_width + bottom_width) / 2
                        untilt_height_mean = (right_height + left_height) / 2

                        width_ratio = untilt_width_mean / width_2d
                        height_ratio = untilt_height_mean / height_2d

                        width_2d = (xmax - xmin) * width_ratio
                        height_2d = (ymax - ymin) * height_ratio
                        draw_2d_box(img, ct_xy_coord_dst[0]- width_2d/2 , ct_xy_coord_dst[1] - height_2d/2,
                                    ct_xy_coord_dst[0] + width_2d/2, ct_xy_coord_dst[1] + height_2d/2 , c=(0, 0, 255))
                        print(" untitlted 2D bbox")
                        print("top_width:" + str(top_width))
                        print("bottom_width:" + str(bottom_width))
                        print("right_height:" + str(right_height))
                        print("left_height:" + str(left_height))
                        cv2.line(img, (int(bbox2D_dst[0][0]), int(bbox2D_dst[0][1])),  (int(bbox2D_dst[1][0]), int(bbox2D_dst[1][1])), (255, 0, 0 ), 2,
                                 lineType=cv2.LINE_AA)
                        cv2.line(img, (int(bbox2D_dst[1][0]), int(bbox2D_dst[1][1])),
                                 (int(bbox2D_dst[3][0]), int(bbox2D_dst[3][1])), (255, 0, 0), 2,
                                 lineType=cv2.LINE_AA)
                        cv2.line(img, (int(bbox2D_dst[2][0]), int(bbox2D_dst[2][1])),
                                 (int(bbox2D_dst[3][0]), int(bbox2D_dst[3][1])), (255, 0, 0), 2,
                                 lineType=cv2.LINE_AA)
                        cv2.line(img, (int(bbox2D_dst[2][0]), int(bbox2D_dst[2][1])),
                                 (int(bbox2D_dst[0][0]), int(bbox2D_dst[0][1])), (255, 0, 0), 2,
                                 lineType=cv2.LINE_AA)
                        cv2.imshow("img",img)
                        cv2.waitKey()
                        #
                        # cv2.line(img2, (int(bbox2D_coord_dst[0][0]), int(bbox2D_coord_dst[0][1])),  (int(bbox2D_coord_dst[1][0]), int(bbox2D_coord_dst[1][1])), (255, 0, 0 ), 2,
                        #          lineType=cv2.LINE_AA)
                        # cv2.line(img2, (int(bbox2D_coord_dst[1][0]), int(bbox2D_coord_dst[1][1])),
                        #          (int(bbox2D_coord_dst[3][0]), int(bbox2D_coord_dst[3][1])), (255, 0, 0), 2,
                        #          lineType=cv2.LINE_AA)
                        # cv2.line(img2, (int(bbox2D_coord_dst[2][0]), int(bbox2D_coord_dst[2][1])),
                        #          (int(bbox2D_coord_dst[3][0]), int(bbox2D_coord_dst[3][1])), (255, 0, 0), 2,
                        #          lineType=cv2.LINE_AA)
                        # cv2.line(img2, (int(bbox2D_coord_dst[2][0]), int(bbox2D_coord_dst[2][1])),
                        #          (int(bbox2D_coord_dst[0][0]), int(bbox2D_coord_dst[0][1])), (255, 0, 0), 2,
                        #          lineType=cv2.LINE_AA)


                        # cv2.circle(img2, (int(bbox2D_coord_dst_ct_x), int(bbox2D_coord_dst_ct_y)), 5, (255, 0, 0), -1)

                        # draw_2d_box(img, bbox2D_coord_dst[0][0], bbox2D_coord_dst[0][1], bbox2D_coord_dst[3][0], bbox2D_coord_dst[3][1], c= (255, 0, 0  ))



                        # draw_2d_box(img, bbox2D_coord_dst[0][0], bbox2D_coord_dst[0][1], bbox2D_coord_dst[3][0], bbox2D_coord_dst[3][1], c= (255, 0, 0  ))
                        # draw_2d_box(img2, bbox2D_coord_dst_ct_x - width_2d/2 , bbox2D_coord_dst_ct_y - height_2d/2,
                        #             bbox2D_coord_dst_ct_x + width_2d/2, bbox2D_coord_dst_ct_y + height_2d/2 , c=(255, 0, 0))



                        # orientation
                        orientation = items[6][0]
                        theta = math.atan2(orientation[1], orientation[0])
                        print(theta, str(theta * 180 / math.pi)[:6])

                        c, s = np.cos(theta), np.sin(theta)
                        # R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
                        # rot_inds = np.argsort(-abs(R[0, :]))
                        # rot_mat =R[:, rot_inds]
                        if (-abs(c) > -abs(s)):
                            theta_real = math.atan2(c, -s)  # (y/x)
                            print(theta_real, str(theta_real * 180 / math.pi)[:6])
                        else:
                            theta_real = theta
                        c_real, s_real = np.cos(theta_real), np.sin(theta_real)

                        # alpha
                        alpha = theta - np.arctan2(ct_x - new_K[0][2], new_K[0][0])
                        if alpha > np.pi:
                            alpha -= 2 * np.pi
                        if alpha < -np.pi:
                            alpha += 2 * np.pi

                        centroid = items[2]
                        inds = np.argsort(-abs(items[0][:, 0]))
                        coeffs = items[1][0, inds]
                        dim = [i * 2 for i in coeffs]
                        rot_mat = np.array([[c_real, -s_real, 0], [s_real, c_real, 0], [0, 0, 1]], dtype=np.float32)

                        # corners_3d = compute_3d_box(inp, dim, centroid, rot_mat, Rtilt)
                        # new_K[0][2] = img_width/2
                        # new_K[1][2] = img_height/2


                        print("centroid")

                        centroid_cam = [centroid[0][i] for i in [0, 2, 1]]
                        centroid_cam[1] = -centroid_cam[1]
                        centroid_3d = project_to_image(np.array([centroid_cam], dtype = np.float32), new_K)
                        cv2.circle(img,(int(centroid_3d[0][0]), int(centroid_3d[0][1])), 5, (0, 0, 255), -1)
                        #3d centroid unTilted
                        # index_3d_src = int(centroid_3d[0][0] + centroid_3d[0][1] * img_width)
                        # index_3d_dst_coord =[x2[index_3d_src], y2[index_3d_src]]
                        # index_3d_dst = int(x2[index_3d_dst_coord[0]] + y2[index_3d_dst_coord[1]]*width)
                        # cv2.circle(img,(int(index_3d_dst_coord[0]), int(index_3d_dst_coord[1])), 5, (255, 255, 255), -1)

                        # bbox3D_centroid_dst = [x3[index_3d], y3[index_3d], z3[index_3d]]
                        # corners_3d = compute_3d_box_untilted(dim, bbox3D_centroid_dst, rot_mat)
                        #3d bbox compute
                        corners_3d = compute_3d_box(img, dim, centroid, rot_mat, Rtilt)
                        box_2d = project_to_image(corners_3d, new_K)
                        draw_box_3d(img, box_2d, (128, 128, 128))
                        # draw_box_3d(img2, box_2d, (128, 128, 128))
                        #project시킨 bbox의 center기준
                        # draw_2d_box(img, centroid_3d[0][0] - width_2d/2 , centroid_3d[0][1] - height_2d/2, centroid_3d[0][0] + width_2d/2 , centroid_3d[0][1] + height_2d/2, c=(0, 0, 255))

                        xmin_projected, xmax_projected, ymin_projected, ymax_projected = project_3d_bbox(box_2d)
                        # print("width, height")
                        # print(width, height)
                        # print
                        # print("bbox2D")
                        # print(xmin_projected,ymin_projected, xmax_projected, ymax_projected)
                        if xmin_projected < 0:
                            xmin_projected = 0
                        if ymin_projected < 0:
                            ymin_projected = 0
                        if xmax_projected > width:
                            xmax_projected = width-1
                        if ymax_projected > height:
                            ymax_projected = height-1
                        # print(xmin_projected,ymin_projected, xmax_projected, ymax_projected)

                        # cv2.line(img, (int(xmin_projected), int(ymin_projected)), (int(xmax_projected), int(ymin_projected)), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                        # cv2.line(img, (int(xmax_projected), int(ymin_projected)), (int(xmax_projected), int(ymax_projected)), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                        # cv2.line(img, (int(xmin_projected), int(ymax_projected)), (int(xmin_projected), int(ymin_projected)), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                        # cv2.line(img, (int(xmax_projected), int(ymax_projected)), (int(xmin_projected), int(ymax_projected)), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                        ct_2d_x = (xmin_projected + xmax_projected) /2
                        ct_2d_y = (ymin_projected + ymax_projected) /2
                        # cv2.circle(img,(int(ct_2d_x), int(ct_2d_y)), 5, (0, 255, 255), -1)
                        label_list = [label,
                                      centroid[0][0], centroid[0][1], centroid[0][2],
                                      coeffs[0], coeffs[1], coeffs[2],
                                      theta,
                                      int(ct_xy_coord_dst[0]), int(ct_xy_coord_dst[1]), bbox2D[2], bbox2D[3],
                                      alpha
                                      ]
                        label_lists.append(label_list[:])


            if len(label_lists)== 0:
                continue
            # fp = open('../data/SUNRGBD/labelIndexing_untilted/'+'{}.txt'.format(image_index), 'w')
            # for j in range(len(label_lists)):
            #     fp.write(str(label_lists[j]) + '\n')
            # fp.close()
            label_lists.clear()

            # cv2.imshow("image", img)
            # cv2.imshow("image2", img2)
            # cv2.imshow("image3", img3)
            # cv2.imshow("image4", img4)
            #
            # cv2.waitKey(0)
            # cv2.imshow("ori image", color_raw)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    else:
        continue
    pass


#bilinear_interpolate
# x0 = np.floor(x2[i]).astype(int)
# x1 = x0 + 1
# y0 = np.floor(y2[i]).astype(int)
# y1 = y0 + 1
# x0 = np.clip(x0, 0, color_raw.shape[1] - 1)
# x1 = np.clip(x1, 0, color_raw.shape[1] - 1)
# y0 = np.clip(y0, 0, color_raw.shape[0] - 1)
# y1 = np.clip(y1, 0, color_raw.shape[0] - 1)
#
# Ia = color_raw[y0, x0]
# Ib = color_raw[y1, x0]
# Ic = color_raw[y0, x1]
# Id = color_raw[y1, x1]
#
# wa = (x1 - x2[i]) * (y1 - y2[i])
# wb = (x1 - x2[i]) * (y2[i] - y0)
# wc = (x2[i] - x0) * (y1 - y2[i])
# wd = (x2[i] - x0) * (y2[i] - y0)
#
# value = wa * Ia + wb * Ib + wc * Ic + wd * Id
# x_int = x2[i].astype(int)
# y_int = y2[i].astype(int)
# img.itemset(y_int, x_int, 0, value[0])
# img.itemset(y_int, x_int, 1, value[1])
# img.itemset(y_int, x_int, 2, value[2])



# for h in range(0, img_height):
#     h = np.asarray(x2)
#     for w in range(0, img_width):
#         for c in range (0, img_channel):
#             img[h,w,c] = bilinear_interpolate(img, )

        # bbox2D coord unproject to 3d
        # bbox2D_coord = [[xmin, ymin, depthInpaint[ymin][xmin]],
        #                 [xmax, ymin, depthInpaint[ymin][xmax]],
        #                 [xmin, ymax, depthInpaint[ymax][xmin]],
        #                 [xmax, ymax, depthInpaint[ymax][xmax]]]
        #
        #                 bbox2D_coord = np.array(bbox2D_coord)  # (4,3)
        # bbox2D_coord = np.transpose(bbox2D_coord)  # (3,4)
        # z = bbox2D_coord[2]
        # x = (bbox2D_coord[0] - new_K[0, 2]) * z / new_K[0, 0]
        # y = (bbox2D_coord[1] - new_K[1, 2]) * z / new_K[1, 1]
        # bbox2D_coord_unprojected = np.array([x, y, z], dtype=np.float32)  # (3,4)
        # bbox2D_coord_unprojected = np.transpose(bbox2D_coord_unprojected)  # (4,3)
        # # cam → world
        # sun_permutation = [0, 2, 1]
        # index = np.argsort(sun_permutation)
        # bbox2D_coord_unprojected = bbox2D_coord_unprojected[:, index]  # (4,3)
        # bbox2D_coord_unprojected[:, 2] *= -1
        # bbox2D_coord_unprojected_untilt = np.dot(Rtilt, np.transpose(bbox2D_coord_unprojected))
        # bbox2D_coord_tilted = project_to_image(np.transpose(bbox2D_coord_unprojected_untilt), new_K)
from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys
# from utils.image import get_affine_transform
# from utils.ddd_utils import compute_box_3d, project_to_image, project_to_image_sun2, project_to_image_sun, alpha2rot_y, compute_box_3d_sun, compute_box_3d_sun_2, compute_box_3d_sun_3,compute_box_3d_sun_4,compute_box_3d_sun_5,compute_box_3d_sun_6
# from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d, draw_box_3d_sun
# from utils.ddd_utils import rot_y2alpha
from numpy.linalg import inv
import os
# np.set_printoptions(threshold=sys.maxsize)
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

    # corners_3d = np.dot(np.transpose(Rtilt), corners_3d)

    # camera_coordinate
    corners_3d = corners_3d.transpose(1, 0)
    sun_permutation = [0, 2, 1]
    index = np.argsort(sun_permutation)
    corners_3d = corners_3d[:, index]
    corners_3d[:, 1] *= -1
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
    xmin_projected = box_2d.transpose()[0].min()
    xmax_projected = box_2d.transpose()[0].max()
    ymin_projected = box_2d.transpose()[1].min()
    ymax_projected = box_2d.transpose()[1].max()

    return xmin_projected, xmax_projected, ymin_projected, ymax_projected
fig = pyplot.figure()
ax = Axes3D(fig)

mat_file = io.loadmat('/home/user/SUNRGBD/SUNRGBDMeta.mat')
# mat_file = io.loadmat('C:/SUNRGBDMeta/SUNRGBDMeta.mat')
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
            ## + image,size
            1255,
            1256,
            4088,
            4089,
            4279,
            4280,
            4766,
            8049,
            8682,
            8832,
            8833,
            8834,
            8835,
            9585,
            10197,
            10198,
            3653,
            #+label problem
            929,
            1233
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

img_error = [
    8049,
    8834,
    9182,
    9270,
    9565,
    9585,
    9586,
    10078,
    10197,
    10198,
]
for range_index in range(4) :
    data = mat_file.popitem()
    if(data[0] == 'SUNRGBDMeta'):
        for i in range(9586, 5000, -1):
            image_index = i
            # image_index = 10196
            #원하는 iindex
            # image_ind/ex = 14
            if i in skip_img:
                continue
            if i in img_error:
                continue
            # path = data[1][0][image_index][5][0]
            # if '//' in path:
            #     path = path.replace('//', '/')
            # path = path.split('/')
            # out_str = ''
            # for i in range(0, len(path)-2):
            #     out_str = out_str + path[i]
            #     out_str += '/'
            #
            # out_str = out_str + 'depth_bfx/'
            # print(out_str)
            # file_list = os.listdir(out_str)
            # print(file_list)
            # out_str = out_str + file_list[-1]

            # color_raw = cv2.imread(data[1][0][image_index][5][0], cv2.COLOR_BGR2RGB)
            # color_raw = cv2.imread(data[1][0][image_index][5][0], cv2.COLOR_RGB2BGR)
            # depth_raw_ori = cv2.imread(data[1][0][image_inde ][4][0], -1)
            # depth_raw = cv2.imread(out_str, -1)

            image_path = data[1][0][image_index][5][0]
            depth_path = data[1][0][image_index][4][0]

            depth_bfx_path = ""
            for i in range(1, len(data[1][0][image_index][4][0])):
                # print(str(-i))
                # print(data[1][0][image_index][4][0][-i])
                if data[1][0][image_index][4][0][-i] is "/":
                    depth_bfx_path = data[1][0][image_index][4][0][:-i]+"_bfx"+ data[1][0][image_index][4][0][-i:]
                    # img_index = i
                    break
            print("img index : " + str(image_index))
            print("depth_bfx_path : " + str(depth_bfx_path))

            color_raw = cv2.imread(image_path, cv2.COLOR_RGB2BGR)
            depth_raw = cv2.imread(depth_bfx_path, -1)

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
            for row in depthInpaint :
                for ele in row :
                    ele = 8 if ele > 8 else ele
                    pass
                pass

            K = np.array(data[1][0][i][3])
            new_K = np.zeros((K.shape[0], K.shape[1] + 1))
            new_K[:, :-1] = K  #(3,3)은 원래데이터            cx = K[0][2]
            cx = K[0][2]
            cy = K[1][2]
            fx = K[0][0]
            fy = K[1][1]
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

            x_under = np.floor(x2).astype(int)
            x_upper = x_under + 1
            y_under = np.floor(y2).astype(int)
            y_upper = y_under + 1

            x2 -= i_xmin
            i_xmax -= i_xmin
            y2 -= i_ymin
            i_ymax -= i_ymin
            # if(i_xmin < 0):
            #     x2 += abs(i_xmin)
            #     i_xmax = i_xmax + abs(i_xmin)
            #
            # if (i_ymin < 0):
            #     y2 += abs(i_ymin)
            #     i_ymax = i_ymax + abs(i_ymin)

            x2 = x2.astype(int)
            y2 = y2.astype(int)


            #x2 & y2 val : pixel position ex. wxh = 730 x 530 = 386900
            #print(x2_minus, y2_minus
            # x2_minus = [(x2[i], y2[i]) for i in range(len(x2)) if x2[i] < 0]
            # print(x2_minus)
            # print("y2_minus")
            # print(len(x2), print(len(y2)))
            # print(y2[i] for i in range(len(y2)) if y2[i] < 0)
            # img = np.zeros((int(ymax)+1, int(xmax)+1, 3), np.uint8)
            # img2 = np.zeros((int(ymax)+1, int(xmax)+1, 3), np.uint8)
            img3 = np.zeros((int(i_ymax)+1, int(i_xmax)+1, 3), np.uint8)

            img_height, img_width, img_channel = img3.shape

            #forward warping

            c = np.array([img3.shape[1] / 2., img3.shape[0] / 2.])  # width, height순으로 이미지의 center point를 구함.

            bbox2D_coord_index = [0,
                                  (width - 1),
                                  (height - 1) * width,
                                  (height - 1) * width + (width - 1)
                                  ]

            ind_coord = [[0, 0],  # (x_min, y_max)
                         [width - 1, 0],  # (x_min, y_min)
                         [0, height - 1],  # (x_max, ymin)
                         [width - 1, height - 1]
                         ]
            bbox2D_coord_src = ind_coord
            bbox2D_coord_dst = [[x2[i], y2[i]] for i in bbox2D_coord_index]

            print(bbox2D_coord_index)


            # print(bbo/x2D_coord_dst)

            projection_matrix = np.zeros((8, 8))  # 나중에 inverse취한후
            # coord_matrix = np.zeros((8,1)) #얘랑 곱해줘야함
            coord_matrix = np.array([bbox2D_coord_dst[0][0], bbox2D_coord_dst[0][1],
                                     bbox2D_coord_dst[1][0], bbox2D_coord_dst[1][1],
                                     bbox2D_coord_dst[2][0], bbox2D_coord_dst[2][1],
                                     bbox2D_coord_dst[3][0], bbox2D_coord_dst[3][1]
                                     ])

            projection_matrix[0][0] = bbox2D_coord_src[0][0]  # x1
            projection_matrix[0][1] = bbox2D_coord_src[0][1]  # y1
            projection_matrix[0][2] = 1.0
            projection_matrix[0][6] = -1 * bbox2D_coord_dst[0][0] * bbox2D_coord_src[0][0]  # -x1'x1
            projection_matrix[0][7] = -1 * bbox2D_coord_dst[0][0] * bbox2D_coord_src[0][1]  # -x1'y1

            projection_matrix[1][3] = bbox2D_coord_src[0][0]  # x1
            projection_matrix[1][4] = bbox2D_coord_src[0][1]  # y1
            projection_matrix[1][5] = 1.0
            projection_matrix[1][6] = -1 * bbox2D_coord_dst[0][1] * bbox2D_coord_src[0][0]  # -y1'x1
            projection_matrix[1][7] = -1 * bbox2D_coord_dst[0][1] * bbox2D_coord_src[0][1]  # -y1'y1

            projection_matrix[2][0] = bbox2D_coord_src[1][0]  # x2
            projection_matrix[2][1] = bbox2D_coord_src[1][1]  # y2
            projection_matrix[2][2] = 1.0
            projection_matrix[2][6] = -1 * bbox2D_coord_dst[1][0] * bbox2D_coord_src[1][0]  # -x2'x2
            projection_matrix[2][7] = -1 * bbox2D_coord_dst[1][0] * bbox2D_coord_src[1][1]  # -x2'y2

            projection_matrix[3][3] = bbox2D_coord_src[1][0]
            projection_matrix[3][4] = bbox2D_coord_src[1][1]
            projection_matrix[3][5] = 1.0
            projection_matrix[3][6] = -1 * bbox2D_coord_dst[1][1] * bbox2D_coord_src[1][0]
            projection_matrix[3][7] = -1 * bbox2D_coord_dst[1][1] * bbox2D_coord_src[1][1]

            projection_matrix[4][0] = bbox2D_coord_src[2][0]
            projection_matrix[4][1] = bbox2D_coord_src[2][1]
            projection_matrix[4][2] = 1.0
            projection_matrix[4][6] = -1 * bbox2D_coord_src[2][0] * bbox2D_coord_dst[2][0]
            projection_matrix[4][7] = -1 * bbox2D_coord_src[2][1] * bbox2D_coord_dst[2][0]

            projection_matrix[5][3] = bbox2D_coord_src[2][0]
            projection_matrix[5][4] = bbox2D_coord_src[2][1]
            projection_matrix[5][5] = 1.0
            projection_matrix[5][6] = -1 * bbox2D_coord_src[2][0] * bbox2D_coord_dst[2][1]
            projection_matrix[5][7] = -1 * bbox2D_coord_src[2][1] * bbox2D_coord_dst[2][1]

            projection_matrix[6][0] = bbox2D_coord_src[3][0]
            projection_matrix[6][1] = bbox2D_coord_src[3][1]
            projection_matrix[6][2] = 1.0
            projection_matrix[6][6] = -1 * bbox2D_coord_src[3][0] * bbox2D_coord_dst[3][0]
            projection_matrix[6][7] = -1 * bbox2D_coord_src[3][1] * bbox2D_coord_dst[3][0]

            projection_matrix[7][3] = bbox2D_coord_src[3][0]
            projection_matrix[7][4] = bbox2D_coord_src[3][1]
            projection_matrix[7][5] = 1.0
            projection_matrix[7][6] = -1 * bbox2D_coord_src[3][0] * bbox2D_coord_dst[3][1]
            projection_matrix[7][7] = -1 * bbox2D_coord_src[3][1] * bbox2D_coord_dst[3][1]

            inv_projection_matrix = inv(projection_matrix)
            perspective_matrix = np.dot(inv_projection_matrix, coord_matrix)
            perspective_matrix = perspective_matrix.tolist()
            perspective_matrix_3d = np.ones((3, 3))
            perspective_matrix_3d[0][0] = perspective_matrix[0]
            perspective_matrix_3d[0][1] = perspective_matrix[1]
            perspective_matrix_3d[0][2] = perspective_matrix[2]
            perspective_matrix_3d[1][0] = perspective_matrix[3]
            perspective_matrix_3d[1][1] = perspective_matrix[4]
            perspective_matrix_3d[1][2] = perspective_matrix[5]
            perspective_matrix_3d[2][0] = perspective_matrix[6]
            perspective_matrix_3d[2][1] = perspective_matrix[7]

            inv_perspective_matrix_3d = inv(perspective_matrix_3d)

            for h in range(0, img_height):
                for w in range(0, img_width):
                    w_coord = inv_perspective_matrix_3d[2][0] * w + inv_perspective_matrix_3d[2][1] * h + inv_perspective_matrix_3d[2][2]
                    # only backwarping
                    x_coord = (inv_perspective_matrix_3d[0][0] * w + inv_perspective_matrix_3d[0][1] * h +
                               inv_perspective_matrix_3d[0][2]) / w_coord
                    y_coord = (inv_perspective_matrix_3d[1][0] * w + inv_perspective_matrix_3d[1][1] * h +
                               inv_perspective_matrix_3d[1][2]) / w_coord

                    wx_1 = x_coord - math.floor(x_coord)
                    wx_0 = 1.0 - wx_1

                    wy_1 = y_coord - math.floor(y_coord)
                    wy_0 = 1.0 - wy_1

                    x_coord_int = int(math.floor(x_coord))
                    y_coord_int = int(math.floor(y_coord))
                    pixel_b, pixel_g, pixel_r, ratio = 0, 0, 0, 0

                    if bound_check(x_coord_int, y_coord_int, width, height):
                        pixel_b += wx_0 * wy_0 * color_raw[y_coord_int][x_coord_int][0]
                        pixel_g += wx_0 * wy_0 * color_raw[y_coord_int][x_coord_int][1]
                        pixel_r += wx_0 * wy_0 * color_raw[y_coord_int][x_coord_int][2]
                        ratio += wx_0 * wy_0

                    if bound_check(x_coord_int + 1, y_coord_int, width, height):
                        pixel_b += wx_1 * wy_0 * color_raw[y_coord_int][x_coord_int + 1][0]
                        pixel_g += wx_1 * wy_0 * color_raw[y_coord_int][x_coord_int + 1][1]
                        pixel_r += wx_1 * wy_0 * color_raw[y_coord_int][x_coord_int + 1][2]
                        ratio += wx_1 * wy_0

                    if bound_check(x_coord_int, y_coord_int + 1, width, height):
                        pixel_b += wx_0 * wy_1 * color_raw[y_coord_int + 1][x_coord_int][0]
                        pixel_g += wx_0 * wy_1 * color_raw[y_coord_int + 1][x_coord_int][1]
                        pixel_r += wx_0 * wy_1 * color_raw[y_coord_int + 1][x_coord_int][2]
                        ratio += wx_0 * wy_1

                    if bound_check(x_coord_int + 1, y_coord_int + 1, width, height):
                        pixel_b += wx_1 * wy_1 * color_raw[y_coord_int + 1][x_coord_int + 1][0]
                        pixel_g += wx_1 * wy_1 * color_raw[y_coord_int + 1][x_coord_int + 1][1]
                        pixel_r += wx_1 * wy_1 * color_raw[y_coord_int + 1][x_coord_int + 1][2]
                        ratio += wx_1 * wy_1
                    if (ratio == 0):
                        ratio = 0.000000001
                    # print(ratio)
                    pixel_b = math.floor(pixel_b / ratio + 0.5)
                    pixel_g = math.floor(pixel_g / ratio + 0.5)
                    pixel_r = math.floor(pixel_r / ratio + 0.5)

                    img3.itemset(h, w, 0, pixel_b)
                    img3.itemset(h, w, 1, pixel_g)
                    img3.itemset(h, w, 2, pixel_r)



            # for groundtruth3DBB in data[1][0][image_index][1]:
            #     img_flag =False
            #     for items in groundtruth3DBB:
            #         bbox2DFlag = items[7].size
            #         if (bbox2DFlag >= 1):
            #             label = items[3][0]
            #             if label not in class_final:
            #                 continue
            #
            #             #ground_truth_2d_bbox(green)
            #             bbox2D = items[7][0]
            #             xmin = bbox2D[0]
            #             ymin = bbox2D[1]
            #             xmax = int(bbox2D[0]) + int(bbox2D[2]) if int(bbox2D[0]) + int(bbox2D[2]) < width - 1 else width - 2
            #             ymax = int(bbox2D[1]) + int(bbox2D[3]) if int(bbox2D[1]) + int(bbox2D[3]) < height - 1 else height - 2
            #             width_2d = xmax - xmin
            #             height_2d = ymax - ymin
            #
            #             ct_x = xmin + bbox2D[2] / 2
            #             ct_y = ymin + bbox2D[3] / 2
            #             ct_x_int = int(ct_x)
            #             ct_y_int = int(ct_y)
            #             #draw ground truth bbox2D(green)
            #             # draw_2d_box(img, xmin, ymin, xmax, ymax)
            #             '''
            #             # backwarping
            #             '''


            cv2.imwrite("/home/user/PycharmProjects/imgs/untilted_GW/"+str(image_index)+".jpg", img3)


            # cv2.imshow("image3", img3)
            # cv2.waitKey(0)
            cv2.imwrite('C:\\obj_detection\\CenterNet-master\\CenterNet-master\\data\\SUNRGBD\\test\\ '+str(image_index)+'.jpg',img3)

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
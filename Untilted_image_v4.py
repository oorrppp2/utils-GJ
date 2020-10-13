from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
from numpy.linalg import inv
import math
import sys

# np.set_printoptions(threshold=sys.maxsize)


fig = pyplot.figure()
ax = Axes3D(fig)

mat_file = io.loadmat('/home/user/SUNRGBD/SUNRGBDMeta.mat')

def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1) #homo_coord를 만드는 과정: (x,y,z,1)로
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  #normalized image plnae에서의 카메라좌표상의 위치를 project시킨 상황.
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]

  # import pdb; pdb.set_trace()
  return pts_2d

def draw_2d_box(color_raw, xmin, ymin, xmax, ymax):
    cv2.line(color_raw, (int(xmin), int(ymin)), (int(xmax), int(ymin)), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.line(color_raw, (int(xmax), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.line(color_raw, (int(xmax), int(ymax)), (int(xmin), int(ymax)), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.line(color_raw, (int(xmin), int(ymax)), (int(xmin), int(ymin)), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    ct_x = (xmin + xmax) / 2
    ct_y = (ymin + ymax) / 2
    cv2.circle(color_raw, (int(ct_x), int(ct_y)), 3, (0, 255, 0), -1)

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

def bound_check(x_boundary, y_boundary, width, height):
    if (x_boundary < 0 or x_boundary >= width  or y_boundary < 0 or y_boundary >= height):
        return 0
    return 1

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
      cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
               (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), c, 2, lineType=cv2.LINE_AA)

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
    # image_index = 5000
    for image_index in range(8050,10335):
        if(data[0] == 'SUNRGBDMeta'):
            # image_index = 60

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
            print("depth_bfx_path : " + str(depth_bfx_path))

            color_raw = cv2.imread(image_path, cv2.COLOR_RGB2BGR)
            depth_raw = cv2.imread(depth_bfx_path, -1)

            height, width, channel = color_raw.shape

            rgb = np.reshape(color_raw, (len(color_raw)*len(color_raw[0]), 3))
            rgb = rgb.astype(np.uint8)
            # rgb = rgb.astype("float32")
            # rgb = rgb / 255

            # print(rgb)

            """
                Make 3d point cloud by using depth_raw
            """
            depthInpaint = (depth_raw>>3) | (depth_raw<<(16-3))
            depthInpaint = depthInpaint.astype("float32")
            depthInpaint = depthInpaint / 1000
            print("depthInpaint min : " + str(np.min(depthInpaint)))

            # print(depthInpaint)
            for row in depthInpaint :
                for ele in row :
                    ele = 8 if ele > 8 else ele
                    pass
                pass

            K = data[1][0][image_index][3]
            new_K = np.zeros((K.shape[0],K.shape[1]+1))
            new_K[:,:-1] = K
            cx = K[0][2]
            cy = K[1][2]
            fx = K[0][0]
            fy = K[1][1]

            range_x = np.arange(0, len(depth_raw[0]))
            range_y = np.arange(0, len(depth_raw))

            x, y = np.meshgrid(range_x, range_y)

            x3 = (x-cx)*depthInpaint*1/fx
            y3 = (y-cy)*depthInpaint*1/fy
            z3 = depthInpaint

            x3 = np.reshape(x3, len(x3)*len(x3[0]))
            y3 = np.reshape(y3, len(y3)*len(y3[0]))
            z3 = np.reshape(z3, len(z3)*len(z3[0]))

            pointsMat = np.vstack((x3,z3,-y3))

            # print("shape : " + str(pointsMat.shape))
            # print(pointsMat[0])

            # remove nan
            nan_index = []
            for i in range(len(x3)):
                # if x3[i] != 0 or y3[i] != 0 or z3[i] != 0:
                if x3[i] == 0 and y3[i] == 0 and z3[i] == 0:
                    # nan_index.append(i)
                    # nan_flag[i] = True
                    # print(pointsMat[:,i])
                    pass
                pass
            pointsMat = np.delete(pointsMat, nan_index, axis=1)
            rgb = np.delete(rgb, nan_index, axis=0)

            # print("pointMat shape : " + str(pointsMat.shape))

            Rtilt = data[1][0][image_index][2]

            print("* Rtilt *")
            print(Rtilt)
            point3d = Rtilt @ pointsMat

            x3 = point3d[0]
            y3 = -point3d[2]
            z3 = point3d[1]

            x2 = (fx*x3 + cx*z3) / z3
            y2 = (fy*y3 + cy*z3) / z3

            x2 = x2.astype(int)
            y2 = y2.astype(int)
            print(K)

            xmin, xmax, ymin, ymax = 10000, 0, 10000, 0
            for i in range(len(x2)):
                if x2[i] < xmin:
                    xmin = x2[i]
                if x2[i] > xmax:
                    xmax = x2[i]
                if y2[i] < ymin:
                    ymin = y2[i]
                if y2[i] > ymax:
                    ymax = y2[i]
            print(xmin, xmax, ymin, ymax)

            img = np.zeros((int(ymax)+1, int(xmax)+1, 3), np.uint8)

            # bbox2D_coord_index= [(ymin*width)+xmin, (ymin*width)+xmax, (ymax*width)+xmin, (ymax*width)+xmax]
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

            ii = int(height / 2 * width / 2 + width / 2)
            ii_list = [ii]

            # 임의의 4점 용
            bbox2D_coord_src = ind_coord
            bbox2D_coord_dst = [[x2[i], y2[i]] for i in bbox2D_coord_index]

            print(bbox2D_coord_dst)
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

            # # img2에 bwarping
            # # img_height, img_width는 dst의 크기기
            print("=========================================")
            for h in range(0, img_height):
                for w in range(0, img_width):
                    w_coord = inv_perspective_matrix_3d[2][0] * w + inv_perspective_matrix_3d[2][1] * h + inv_perspective_matrix_3d[2][2]
                    # w_coord = perspective_matrix_3d[2][0] * w + perspective_matrix_3d[2][1] * h + 1
                    # print(w_coord)
                    # only backwarping
                    # w_coord = 1
                    x_coord = int(
                        (inv_perspective_matrix_3d[0][0] * w + inv_perspective_matrix_3d[0][1] * h +
                         inv_perspective_matrix_3d[0][2]) / w_coord)
                    y_coord = int(
                        (inv_perspective_matrix_3d[1][0] * w + inv_perspective_matrix_3d[1][1] * h +
                         inv_perspective_matrix_3d[1][2]) / w_coord)
                    # x_coord = int(x_coord *size_compare_ratio[0])
                    # y_coord = int(y_coord * size_compare_ratio[1])

                    if not bound_check(x_coord, y_coord, width, height):
                        continue

                    img2.itemset(h, w, 0, color_raw[y_coord][x_coord][0])
                    img2.itemset(h, w, 1, color_raw[y_coord][x_coord][1])
                    img2.itemset(h, w, 2, color_raw[y_coord][x_coord][2])

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

                    img2.itemset(h, w, 0, pixel_b)
                    img2.itemset(h, w, 1, pixel_g)
                    img2.itemset(h, w, 2, pixel_r)

            # cv2.imshow("image", img)
            # cv2.waitKey(0)
            # cv2.imshow("image_backwarping", img2)
            # cv2.waitKey(0)
            # cv2.imshow("image_forwardwarping", img3)
            # cv2.waitKey(0)

            cv2.imwrite("/home/user/PycharmProjects/imgs/untilted_img/"+str(image_index)+".jpg", img2)

            # cv2.imshow("image_interpolation", img)
            # cv2.waitKey(0)
            # cv2.imshow("ori image", color_raw)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        else:
            continue
        pass


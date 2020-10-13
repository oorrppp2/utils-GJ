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
    image_index = 0
    for image_index in range(10335):
        if(data[0] == 'SUNRGBDMeta'):
            image_index = 60

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

            color_raw = cv2.imread(image_path, cv2.COLOR_RGB2BGR)
            depth_raw = cv2.imread(depth_bfx_path, -1)

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
            # for row in depthInpaint :
            #     for ele in row :
            #         ele = 8 if ele > 8 else ele
            #         pass
            #     pass

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
                    nan_index.append(i)
                    # nan_flag[i] = True
                    # print(pointsMat[:,i])
                    pass
                pass
            pointsMat = np.delete(pointsMat, nan_index, axis=1)
            rgb = np.delete(rgb, nan_index, axis=0)

            Rtilt = data[1][0][image_index][2]

            print("* Rtilt *")
            print(Rtilt)
            point3d = Rtilt @ pointsMat

            x3 = point3d[0]
            y3 = -point3d[2]
            z3 = point3d[1]

            x2 = (fx*x3 + cx*z3) / z3
            y2 = (fy*y3 + cy*z3) / z3

            minus_index = []
            for i in range(len(x2)):
                if x2[i] < 0 or y2[i] < 0: # or x2[i] > len(depth_raw[0]) or y2[i] > len(depth_raw):
                    minus_index.append(i)
            x2 = np.delete(x2, minus_index, axis=0)
            y2 = np.delete(y2, minus_index, axis=0)
            rgb = np.delete(rgb, minus_index, axis=0)

            x2 = x2.astype(int)
            y2 = y2.astype(int)

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

            for i in range(0, len(x2)):
                img.itemset(y2[i], x2[i], 0, rgb[i, 0])
                img.itemset(y2[i], x2[i], 1, rgb[i, 1])
                img.itemset(y2[i], x2[i], 2, rgb[i, 2])

            # c = np.array([img.shape[1] / 2., img.shape[0] / 2.])  # width, height순으로 이미지의 center point를 구함.
            # s = np.array([width, height], dtype=np.int32)  # (448, 448)
            # trans_input = get_affine_transform(c, s, 0, [width, height])
            # inp = cv2.warpAffine(img, trans_input,
            #                      (width, height),
            #                      flags=cv2.INTER_LINEAR)


            for groundtruth3DBB in data[1][0][image_index][1]:
                for items in groundtruth3DBB:
                    bbox2DFlag = items[7].size
                    if (bbox2DFlag >= 1):
                        label = items[3][0]
                        if label not in class_final:
                            continue

                        bbox2D = items[7][0]
                        xmin = bbox2D[0]
                        ymin = bbox2D[1]
                        xmax = int(bbox2D[0]) + int(bbox2D[2]) if bbox2D[0] + bbox2D[2] < xmax - 1 else xmax - 1
                        ymax = int(bbox2D[1]) + int(bbox2D[3]) if bbox2D[1] + bbox2D[3] < ymax - 1 else ymax - 1
                        # print('2D box')
                        # print(xmax, ymax)
                        ct_x = xmin + bbox2D[2] / 2
                        ct_y = ymin + bbox2D[3] / 2
                        draw_2d_box(img, xmin, ymin, xmax, ymax)

                        # orientation
                        orientation = items[6][0]
                        theta = math.atan2(orientation[1], orientation[0])
                        # print(theta, str(theta * 180 / math.pi)[:6])

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

                        centroid = items[2]
                        inds = np.argsort(-abs(items[0][:, 0]))
                        coeffs = items[1][0, inds]
                        dim = [i * 2 for i in coeffs]
                        rot_mat = np.array([[c_real, -s_real, 0], [s_real, c_real, 0], [0, 0, 1]], dtype=np.float32)

                        corners_3d = compute_3d_box(img, dim, centroid, rot_mat, Rtilt)
                        box_2d = project_to_image(corners_3d, new_K)
                        draw_box_3d(img, box_2d, (128, 128, 128))


            cv2.imshow("image", img)
            cv2.waitKey(0)
            cv2.imshow("image2", color_raw)
            cv2.waitKey(0)
            # cv2.imwrite("/home/user/PycharmProjects/imgs/untilted_img/"+str(image_index)+".jpg", img)

            # cv2.imshow("image_interpolation", img)
            # cv2.waitKey(0)
            # cv2.imshow("ori image", color_raw)
            # cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            continue
        pass


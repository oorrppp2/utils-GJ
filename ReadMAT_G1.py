from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys
import numpy.linalg as lin

# np.set_printoptions(threshold=sys.maxsize)

fig = pyplot.figure()
ax = Axes3D(fig)

mat_file = io.loadmat('/home/user/SUNRGBD/SUNRGBDMeta.mat')

"""
    Original CenterNet
"""
def compute_box_3d(dim, location, rotation_y, Rtilt):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  l, w, h = dim[2], dim[1], dim[0]
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0,0,0,0,-h,-h,-h,-h]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners)  #R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
  temp = np.array(location, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1) #camera좌표 (0,0)에서 시작했었으니까, 물체 center point로 평행이동시킴

  Rtilt = np.transpose(Rtilt)

  Rtilt_revised = np.zeros((3,3))
  Rtilt_revised[0,0] = Rtilt[0,0]
  Rtilt_revised[0,1] = -Rtilt[0,2]
  Rtilt_revised[0,2] = Rtilt[0,1]

  Rtilt_revised[1,0] = -Rtilt[2,0]
  Rtilt_revised[1,1] = Rtilt[2,2]
  Rtilt_revised[1,2] = -Rtilt[2,1]

  Rtilt_revised[2,0] = Rtilt[1,0]
  Rtilt_revised[2,1] = -Rtilt[1,2]
  Rtilt_revised[2,2] = Rtilt[1,1]

  corners_3d = np.dot(Rtilt_revised, corners_3d)
  # corners_3d = corners_3d.transpose(1, 0)
  return corners_3d.transpose(1, 0)

def unproject_2d_to_3d(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  # return: 3
  z = depth - P[2, 3]
  x = (pt_2d[0] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
  y = (pt_2d[1] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
  pt_3d = np.array([x, y, z], dtype=np.float32)
  return pt_3d

def draw_box_3d(image, corners, c=(0, 0, 255)):
  face_idx = [[0,1,5,4],
              [1,2,6, 5],
              [2,3,7,6],
              [3,0,4,7]]
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
               (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), c, 2, lineType=cv2.LINE_AA)
      # cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
      #          (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), c, 2, lineType=cv2.LINE_AA)
    if ind_f == 0:
        cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
                 (int(corners[f[2], 0]), int(corners[f[2], 1])), c, 1, lineType=cv2.LINE_AA)
        cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
                 (int(corners[f[3], 0]), int(corners[f[3], 1])), c, 1, lineType=cv2.LINE_AA)
      # cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
      #          (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
      # cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
      #          (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
  return image

"""
    Customized CenterNet
"""
def draw_box_3d_world(image, corners, c=(0, 0, 255)):
  # face_idx = [[7,6,2,3], #앞
  #             [7,4,0,3], #왼
  #             [4,5,1,0],  #뒤
  #             [1,5,6,2]] #오
  face_idx = [[1,2,6,5],
              [2, 3, 7, 6],
              [0,1,5,4], #앞
              [3,0,4,7]] #오
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      #print(corners.shape)
      #print(corners)
      cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
               (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), c, 2, lineType=cv2.LINE_AA)
      if ind_f == 0:  # 암면에 대해서는 대각선으로 표시
        cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
                 (int(corners[f[2], 0]), int(corners[f[2], 1])), c, 1, lineType=cv2.LINE_AA)
        cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
                 (int(corners[f[3], 0]), int(corners[f[3], 1])), c, 1, lineType=cv2.LINE_AA)
      # cv2.imshow(str(image_id), image)
      # cv2.waitKey()
  return image


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

def compute_box_3d_sun_1(dim, location, theta_c, Rtilt):
  c, s = np.cos(theta_c), np.sin(theta_c)
  print("rotation_y:" + str(theta_c) + "각도는" + str(theta_c * 180 / math.pi))

  R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

  #when R matrix가 z중심일 때
  rot_inds = np.argsort(-abs(R[0, :]))
  R =R[:, rot_inds]

  print("after_np.argsort" + str(R))
  rot_coeffs = abs(np.array(dim))
  x = rot_coeffs[0]
  y = rot_coeffs[1]
  z = rot_coeffs[2]
  x_corners = [-x, x, x, -x, -x, x, x, -x]
  y_corners = [y, y, -y, -y, y, y, -y, -y]
  z_corners = [z, z, z, z, -z, -z, -z, -z]
  corners2 = np.array([x_corners, y_corners, z_corners], dtype=np.float32)

  corners_3d = np.dot(R, corners2)  #(3,8)
  temp = np.array(location, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3,1) #(3,8)

  sun_permutation = [0,2,1]
  index = np.argsort(sun_permutation)
  corners_3d = corners_3d[index, :]
  corners_3d[1, :] *= -1

  Rtilt = np.transpose(Rtilt)

  Rtilt_revised = np.zeros((3,3))
  Rtilt_revised[0,0] = Rtilt[0,0]
  Rtilt_revised[0,1] = -Rtilt[0,2]
  Rtilt_revised[0,2] = Rtilt[0,1]

  Rtilt_revised[1,0] = -Rtilt[2,0]
  Rtilt_revised[1,1] = Rtilt[2,2]
  Rtilt_revised[1,2] = -Rtilt[2,1]

  Rtilt_revised[2,0] = Rtilt[1,0]
  Rtilt_revised[2,1] = -Rtilt[1,2]
  Rtilt_revised[2,2] = Rtilt[1,1]

  # corners_3d = np.dot(Rtilt_revised, corners_3d)
  corners_3d = np.dot(Rtilt_revised, corners_3d)
  corners_3d = corners_3d.transpose(1, 0)
  return corners_3d

def compute_box_3d_sun_2(dim, location, theta_c, Rtilt):
  c, s = np.cos(theta_c), np.sin(theta_c)
  print("rotation_y:" + str(theta_c) + "각도는" + str(theta_c * 180 / math.pi))

  R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

  #when R matrix가 z중심일 때
  rot_inds = np.argsort(-abs(R[0, :]))
  R =R[:, rot_inds]

  print("after_np.argsort" + str(R))
  rot_coeffs = abs(np.array(dim))
  x = rot_coeffs[0]
  y = rot_coeffs[1]
  z = rot_coeffs[2]
  x_corners = [-x, x, x, -x, -x, x, x, -x]
  y_corners = [y, y, -y, -y, y, y, -y, -y]
  z_corners = [z, z, z, z, -z, -z, -z, -z]
  corners2 = np.array([x_corners, y_corners, z_corners], dtype=np.float32)

  corners_3d = np.dot(R, corners2)  #(3,8)
  temp = np.array(location, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3,1) #(3,8)

  sun_permutation = [0,2,1]
  index = np.argsort(sun_permutation)
  corners_3d = corners_3d[index, :]
  corners_3d[1, :] *= -1

  Rtilt = np.transpose(Rtilt)

  Rtilt_revised = np.zeros((3,3))
  Rtilt_revised[0,0] = Rtilt[0,0]
  Rtilt_revised[0,1] = -Rtilt[0,2]
  Rtilt_revised[0,2] = Rtilt[0,1]

  Rtilt_revised[1,0] = -Rtilt[2,0]
  Rtilt_revised[1,1] = Rtilt[2,2]
  Rtilt_revised[1,2] = -Rtilt[2,1]

  Rtilt_revised[2,0] = Rtilt[1,0]
  Rtilt_revised[2,1] = -Rtilt[1,2]
  Rtilt_revised[2,2] = Rtilt[1,1]

  corners_3d = np.dot(Rtilt_revised, corners_3d)
  corners_3d = np.dot(Rtilt_revised, corners_3d)

  Rtilt_inverse = lin.inv(Rtilt_revised)
  corners_3d = np.dot(Rtilt_inverse, corners_3d)
  corners_3d = corners_3d.transpose(1, 0)
  return corners_3d

def compute_box_3d_sun_2(dim, location, theta_c, Rtilt):
  c, s = np.cos(theta_c), np.sin(theta_c)
  print("rotation_y:" + str(theta_c) + "각도는" + str(theta_c * 180 / math.pi))

  R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

  #when R matrix가 z중심일 때
  rot_inds = np.argsort(-abs(R[0, :]))
  R =R[:, rot_inds]

  print("after_np.argsort" + str(R))
  rot_coeffs = abs(np.array(dim))
  x = rot_coeffs[0]
  y = rot_coeffs[1]
  z = rot_coeffs[2]
  x_corners = [-x, x, x, -x, -x, x, x, -x]
  y_corners = [y, y, -y, -y, y, y, -y, -y]
  z_corners = [z, z, z, z, -z, -z, -z, -z]
  corners2 = np.array([x_corners, y_corners, z_corners], dtype=np.float32)

  corners_3d = np.dot(R, corners2)  #(3,8)
  temp = np.array(location, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3,1) #(3,8)

  sun_permutation = [0,2,1]
  index = np.argsort(sun_permutation)
  corners_3d = corners_3d[index, :]
  corners_3d[1, :] *= -1

  Rtilt = np.transpose(Rtilt)

  Rtilt_revised = np.zeros((3,3))
  Rtilt_revised[0,0] = Rtilt[0,0]
  Rtilt_revised[0,1] = -Rtilt[0,2]
  Rtilt_revised[0,2] = Rtilt[0,1]

  Rtilt_revised[1,0] = -Rtilt[2,0]
  Rtilt_revised[1,1] = Rtilt[2,2]
  Rtilt_revised[1,2] = -Rtilt[2,1]

  Rtilt_revised[2,0] = Rtilt[1,0]
  Rtilt_revised[2,1] = -Rtilt[1,2]
  Rtilt_revised[2,2] = Rtilt[1,1]

  corners_3d = np.dot(Rtilt_revised, corners_3d)
  corners_3d = np.dot(Rtilt_revised, corners_3d)

  Rtilt_inverse = lin.inv(Rtilt_revised)
  corners_3d = np.dot(Rtilt_inverse, corners_3d)
  corners_3d = corners_3d.transpose(1, 0)
  return corners_3d

for range_index in range(4) :

    data = mat_file.popitem()
    if(data[0] == 'SUNRGBDMeta'):
        """
            data[1][0][index] --> index 번째 data. ( 0 <= index <= 10335 )
            data[1][0][index][index2] --> index 번째 data의 index2 번째 data. Matlab에서 column에 해당. 13개.
        """

        for i in range(10335):
            image_index = i

            print("Img path : ", data[1][0][image_index][5][0])

            # color_raw = cv2.imread(data[1][0][image_index][5][0], cv2.COLOR_BGR2RGB)
            image = cv2.imread(data[1][0][image_index][5][0], cv2.COLOR_RGB2BGR)

            # rgb = np.reshape(color_raw, (len(color_raw)*len(color_raw[0]), 3))
            # rgb = rgb.astype("float32")
            # rgb = rgb / 255

            Rtilt = data[1][0][image_index][2]

            calib = np.zeros((3,4))
            K = data[1][0][image_index][3]
            # np.expand_dims(K,axis=1)
            calib[:,:3] = K

            for groundtruth3DBB in data[1][0][image_index][1]:
                for items in groundtruth3DBB:
                    # 가려진 정도를 의미 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
                    # tmp순서: centroid_x, centroid_y, centroid_z, length, height, width
                    # 방법1) x,z,y수넛로 값을 집어 넣기

                    inds = np.argsort(-abs(items[0][:, 0]))
                    location = items[2][0]
                    dim = abs(items[1][0, inds])

                    # dim_permutation = [1, 0, 2]
                    # dim_index = np.argsort(dim_permutation)
                    # dim = dim[dim_index]
                    #
                    # loc_permutation = [0, 2, 1]
                    # loc_index = np.argsort(loc_permutation)
                    # location = location[loc_index]
                    # location[1] *= -1

                    orientation = items[6][0]
                    rotation_y = math.atan2(orientation[1], orientation[0])

                    bbox2D = items[7][0]
                    xmin = int(bbox2D[0])
                    ymin = int(bbox2D[1])
                    xmax = int(bbox2D[0]) + int(bbox2D[2])

                    ymax = int(bbox2D[1]) + int(bbox2D[3])
                    # x = (bbox2D[0] + bbox2D[2]) / 2
                    # cv2.line(image, (xmin, ymin), (xmax, ymin), (0, 255, 0), 2, lineType=cv2.LINE_AA)
                    # cv2.line(image, (xmax, ymin), (xmax, ymax), (0, 255, 0), 2, lineType=cv2.LINE_AA)
                    # cv2.line(image, (xmin, ymax), (xmin, ymin), (0, 255, 0), 2, lineType=cv2.LINE_AA)
                    # cv2.line(image, (xmax, ymax), (xmin, ymax), (0, 255, 0), 2, lineType=cv2.LINE_AA)
                    # ct_x = xmin + bbox2D[2] / 2
                    # ct_y = ymin + bbox2D[3] / 2
                    # cv2.circle(image, (int(ct_x), int(ct_y)), 10, (0, 255, 0), -1)
                    # end_x = ct_x + np.cos(rotation_y) * 100
                    # end_y = ct_y - np.sin(rotation_y) * 100
                    # cv2.arrowedLine(image, (int(ct_x), int(ct_y)), (int(end_x), int(end_y)), (0, 255, 0), 2)  # 초록

                    box_3d_r = compute_box_3d_sun_1(dim, location, rotation_y, Rtilt)
                    box_3d_w = compute_box_3d_sun_2(dim, location, rotation_y, Rtilt)
                    box_2d_r = project_to_image(box_3d_r, calib)  # 이제 3d bounding box를 image에 투영시킴
                    box_2d_w = project_to_image(box_3d_w, calib)  # 이제 3d bounding box를 image에 투영시킴
                    image = draw_box_3d_world(image, box_2d_r)
                    image = draw_box_3d_world(image, box_2d_w, c=(255,255,255))

                    # box_3d = compute_box_3d(dim, location, rotation_y, Rtilt)
                    # box_2d = project_to_image(box_3d, calib)
                    # print('box_2d', box_2d)
                    # image = draw_box_3d(image, box_2d)
                    pass
                pass
            cv2.imshow(str(image_index), image)
            cv2.waitKey()

    else:
        continue
    pass


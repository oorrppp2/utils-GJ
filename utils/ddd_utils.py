from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.spatial.transform import Rotation as Rot

import numpy as np
import cv2

def compute_box_3d(dim, location, rotation_y):
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
  return corners_3d.transpose(1, 0)

def compute_box_3d_sun(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  #l, w, h = dim[2], dim[1], dim[0]
  l, w, h = dim[0], dim[1], dim[2]
  #w는 세로를 담당
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0,0,0,0,-h,-h,-h,-h]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
  corners = np.array([x_corners, z_corners, y_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners)  #R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
  temp = np.array(location, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1) #camera좌표 (0,0)에서 시작했었으니까, 물체 center point로 평행이동시킴
  return corners_3d.transpose(1, 0)

def compute_box_3d_sun_2(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3

  #현재 dim이랑 location이 x,z,y. length, width, height순으로 설정되어 있는 상태
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  #l, w, h = dim[2], dim[1], dim[0]
  l, w, h = dim[0], dim[1], dim[2]
  #w는 세로를 담당
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  #y_corners = [0,0,0,0,-h,-h,-h,-h]
  #y_corners = [h, h, h, h, 0, 0, 0, 0]
  y_corners = [-h, -h, -h, -h, 0, 0, 0, 0]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
  corners = np.array([x_corners, z_corners, y_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners)  #R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
  temp = np.array(location, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1) #camera좌표 (0,0)에서 시작했었으니까, 물체 center point로 평행이동시킴
  return corners_3d.transpose(1, 0)

def compute_box_3d_sun_3(dim, location, rotation_y, Rtilt_ori):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  # imsi = Rtilt_ori[1,:]
  # Rtilt_ori[1,:] = Rtilt_ori[2,:]
  # Rtilt_ori[2,:] = imsi
  # imsi = Rtilt_ori[:,1]
  # Rtilt_ori[:,1] = Rtilt_ori[:,2]
  # Rtilt_ori[:,2] = imsi

  c, s = np.cos(rotation_y), np.sin(rotation_y)
  #R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  #R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  #R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
  # R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)


  R = np.zeros((3, 3), dtype=np.float32)
  R[0, 0] = c
  R[0, 1] = -s
  R[0, 2] = 0

  R[1, 0] = s
  R[1, 1] = c
  R[1, 2] = 0

  R[2, 0] = 0
  R[2, 1] = 0
  R[2, 2] = 1

  rot_inds = np.argsort(-abs(R[:, 0]))

  R = R[rot_inds, :]


  # R_tilt = np.zeros((3,3))
  # R_tilt[0] = [0.979589, 0.012593, -0.200614]
  # R_tilt[2] = [0.012593, 0.992231, 0.123772]
  # R_tilt[1] = [0.200614, -0.123772, 0.97182]
  #Rtilt_ori = np.array([[0.979589, 0.012593, -0.200614],[0.012593, 0.992231, 0.123772],[0.200614, -0.123772, 0.97182]], dtype=np.float32)

  R_tilt = np.matrix(Rtilt_ori).I
  # imsi = R_tilt[:,1]
  # R_tilt[:,1] = R_tilt[:,2]
  # R_tilt[:,2] = imsi

  #l, w, h = dim[2], dim[1], dim[0]
  l, h, w = dim[0]*2 , dim[1]*2 , dim[2]*2
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [h, h, h, h, 0, 0, 0, 0]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]


  # l, w, h = dim[2], dim[1], dim[0]
  # l, h, w = dim[0] * 2, dim[1] * 2, dim[2] * 2
  # x_corners = [-dim[0], dim[0], dim[0], -dim[0], -dim[0], dim[0], dim[0], -dim[0]]
  # y_corners = [dim[1], dim[1], -dim[1], -dim[1], dim[1], dim[1], -dim[1], -dim[1]]
  # z_corners = [dim[2], dim[2], dim[2], dim[2], -dim[2], -dim[2], -dim[2], -dim[2]]


  # corners = np.array([x_corners, z_corners, y_corners], dtype=np.float32)
  corners = np.array([x_corners, z_corners, y_corners], dtype=np.float32)
  location_sun = [location[0], location[2], location[1]]
  # corners = np.dot(R_tilt, corners)
  # corners = np.dot(Rtilt_ori, corners)
  corners_3d = np.dot(corners.transpose(1,0), R).transpose(1,0)  #R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
  # corners_3d[2] = -corners_3d[2]
  # corners_3d = Rtilt_ori @ corners_3d
  # corners_3d = R_tilt @ corners_3d
  temp = np.array(location_sun, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location_sun, dtype=np.float32).reshape(3, 1) #camera좌표 (0,0)에서 시작했었으니까, 물체 center point로 평행이동시킴
  #corners_3d = np.dot(R_tilt, corners_3d)

  return corners_3d.transpose(1, 0)

def compute_box_3d_sun_4(dim, location, rotation_y, Rtilt_ori):

  c, s = np.cos(rotation_y), np.sin(rotation_y)

  R = np.zeros((3, 3), dtype=np.float32)
  R[0, 0] = c
  R[0, 1] = -s
  R[0, 2] = 0

  R[1, 0] = s
  R[1, 1] = c
  R[1, 2] = 0

  R[2, 0] = 0
  R[2, 1] = 0
  R[2, 2] = 1

  rot_inds = np.argsort(-abs(R[:, 0]))
  R = R[rot_inds, :]

  R_tilt = np.matrix(Rtilt_ori).I
  # imsi = R_tilt[:,1]
  # R_tilt[:,1] = R_tilt[:,2]
  # R_tilt[:,2] = imsi

  # l, h, w = dim[0]*2 , dim[1]*2 , dim[2]*2
  # x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  # y_corners = [h, h, h, h, 0, 0, 0, 0]
  # z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

  x_corners = [-dim[0], dim[0], dim[0], -dim[0], -dim[0], dim[0], dim[0], -dim[0]]
  y_corners = [dim[1], dim[1], -dim[1], -dim[1], dim[1], dim[1], -dim[1], -dim[1]]
  z_corners = [dim[2], dim[2], dim[2], dim[2], -dim[2], -dim[2], -dim[2], -dim[2]]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  location_sun = [location[0], location[1], location[2]]
  # corners = np.dot(R_tilt, corners)
  # corners = np.dot(Rtilt_ori, corners)
  corners_3d = np.dot(corners.transpose(1,0), R).transpose(1,0)  #R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
  # corners_3d[2] = -corners_3d[2]
  # corners_3d = Rtilt_ori @ corners_3d
  print("corners_3d : ")
  print(corners_3d.transpose(1,0))
  print()
  corners_3d = R_tilt @ corners_3d
  print("R_tilt @ corners_3d : ")
  print(corners_3d.transpose(1,0))

  temp = np.array(location_sun, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location_sun, dtype=np.float32).reshape(3, 1) #camera좌표 (0,0)에서 시작했었으니까, 물체 center point로 평행이동시킴
  #corners_3d = np.dot(R_tilt, corners_3d)

  return corners_3d.transpose(1, 0)

def project_to_image_sun(pts_3d, P):
  pts_3d[:, 1] = -pts_3d[:, 1]
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  #P[0][3] = 1000
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1) #homo_coord를 만드는 과정: (x,y,z,1)로
  print("K : ")
  print(P)
  # print("pts_3d : ")
  # print(pts_3d)
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  print("pts_2d : ")
  print(pts_2d)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  print("after pts_2d : ")
  print(pts_2d)
  # import pdb; pdb.set_trace()
  return pts_2d


def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1) #homo_coord를 만드는 과정: (x,y,z,1)로
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  # import pdb; pdb.set_trace()
  return pts_2d

def compute_orientation_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 2 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  orientation_3d = np.array([[0, dim[2]], [0, 0], [0, 0]], dtype=np.float32)
  orientation_3d = np.dot(R, orientation_3d)
  orientation_3d = orientation_3d + \
                   np.array(location, dtype=np.float32).reshape(3, 1)
  return orientation_3d.transpose(1, 0)

def draw_box_3d(image, corners, c=(0, 0, 255)):
  face_idx = [[0,1,5,4], #앞
              [1,2,6, 5], #왼
              [2,3,7,6],  #뒤
              [3,0,4,7]] #오
  # face_idx = [[[2,3,7,6]],
  #              [3,0,4,7],
  #              [[0,1,5,4]],
  #              [1,2,6,5]]
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
               (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), c, 2, lineType=cv2.LINE_AA)
      #cv2.imshow("img", image)
      #cv2.waitKey()
    if ind_f == 0: #암면에 대해서는 대각선으로 표시
      cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
               (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
      cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
               (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
  return image

def draw_box_3d_sun(image, corners, c=(0, 0, 255)):
  face_idx = [[0,1,5,4], #앞
              [1,2,6, 5], #왼
              [2,3,7,6],  #뒤
              [3,0,4,7]] #오
  # face_idx = [[[2,3,7,6]],
  #              [3,0,4,7],
  #              [[0,1,5,4]],
  #              [1,2,6,5]]
  cv2.line(image, (corners[4, 0],corners[7, 0]), (200- corners[4, 1],200 -corners[7, 1]), c, 2, lineType=cv2.LINE_AA)
  cv2.line(image, (corners[7, 0],corners[6, 0]), (corners[7, 1],corners[6, 1]), c, 2, lineType=cv2.LINE_AA)

  # for ind_f in range(3, -1, -1):
  #   f = face_idx[ind_f]
  #   for j in range(4):
  #     cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
  #              (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), c, 2, lineType=cv2.LINE_AA)
  cv2.imshow("img", image)
  cv2.waitKey()
  return image

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

def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, fx)
    if rot_y > np.pi:
      rot_y -= 2 * np.pi
    if rot_y < -np.pi:
      rot_y += 2 * np.pi
    return rot_y

def rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha


def ddd2locrot(center, alpha, dim, depth, calib):
  # single image
  locations = unproject_2d_to_3d(center, depth, calib)
  locations[1] += dim[0] / 2
  rotation_y = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
  return locations, rotation_y

def project_3d_bbox(location, dim, rotation_y, calib):
  box_3d = compute_box_3d(dim, location, rotation_y)
  box_2d = project_to_image(box_3d, calib)
  return box_2d


if __name__ == '__main__':
  calib = np.array(
    [[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02, 4.575831000000e+01],
     [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03]],
    dtype=np.float32)
  alpha = -0.20
  tl = np.array([712.40, 143.00], dtype=np.float32)
  br = np.array([810.73, 307.92], dtype=np.float32)
  ct = (tl + br) / 2
  rotation_y = 0.01
  print('alpha2rot_y', alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0]))
  print('rotation_y', rotation_y)
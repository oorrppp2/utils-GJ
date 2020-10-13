from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys

# np.set_printoptions(threshold=sys.maxsize)

def flip_towards_viewer(normals, points):
    mat = np.matlib.repmat(np.sqrt(np.sum(points*points, 1)), 3, 1)
    points = points / mat
    # print(points)
    proj = np.sum(points * normals, 1)
    flip = proj > 0
    normals[flip, :] = -normals[flip, :]
    return normals

fig = pyplot.figure()
ax = Axes3D(fig)

mat_file = io.loadmat('/home/user/SUNRGBD/SUNRGBDMeta.mat')


for range_index in range(4) :

    data = mat_file.popitem()
    if(data[0] == 'SUNRGBDMeta'):

        # for i in range(10335):
        #     d = data[1][0][i]
        #     for j in range(13):
        #
        #         """
        #             data[1][0][image_index][index]
        #             0. sequenceName     (string)
        #             1. groundtruth3DBB  (9 element struct)
        #             2. Rtilt            (3x3 double)
        #             3. K                (3x3 double)
        #             4. depthpath        (string)
        #             5. rgbpath          (string)
        #             6. anno_extrinsics  (3x3 double)
        #             7. depthname        (string)
        #             8. rgbname          (string)
        #             9. sensorType       (string : kv1 / kv2 / realsense / xtion)
        #             10. valid           (전부 1.)
        #             11. gtCorner3D      (3xN double, 없는것도 있음)
        #             12. groundtruth2DBB (4 element struct)
        #         """
        # """
        #     data[1][0][index] --> index 번째 data. ( 0 <= index <= 10335 )
        #     data[1][0][index][index2] --> index 번째 data의 index2 번째 data. Matlab에서 column에 해당. 13개.
        # """

        image_index = 3

        for groundtruth3DBB in data[1][0][image_index][1]:
            for items in groundtruth3DBB:

                """
                    items = data[1][0][image_index][1(groundtruth3DBB)]
                    items[index]
                    0. basis            (3x3 double)
                    1. coeffs           (1x3 double)
                    2. centroid         (1x3 double)
                    3. classname        (string)
                    4. labelname        (?)
                    5. sequenceName     (string)
                    6. orientation      (1x3 double)
                    7. gtBb2D           (1x4 double)
                    8. label            ()
                """

                corners = np.zeros((8,3))
                basis_ori = items[0]

                label = items[3][0]
                inds = np.argsort(-abs(items[0][:, 0]))

                basis = items[0][inds, :]
                coeffs = items[1][0, inds]

                inds = np.argsort(-abs(basis[1:, 1]))

                centroid = items[2]
                basis = flip_towards_viewer(basis, np.matlib.repmat(centroid, 3, 1))
                coeffs = abs(coeffs)

                orientation = items[6][0]

                theta = math.atan2(orientation[1], orientation[0])
                theta_s = str(theta*180.0/math.pi)[:6]
                # dtheta = math.degrees(theta)
                dtheta = theta*180/3.141592

                print("label : " , label , " //", "theta : ", dtheta)
                # print()

                pass
            pass

        # color_raw = cv2.imread(data[1][0][image_index][5][0], cv2.COLOR_RGB2BGR)
        # cv2.imshow(color_raw)
        # cv2.waitKey(0)
    else:
        continue
    pass


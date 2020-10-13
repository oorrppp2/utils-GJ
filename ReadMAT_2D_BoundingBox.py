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
        """
            data[1][0][index] --> index 번째 data. ( 0 <= index <= 10335 )
            data[1][0][index][index2] --> index 번째 data의 index2 번째 data. Matlab에서 column에 해당. 13개.
        """

        # image_index = 422
        # image_index = 598

        # image_index = 1599
        image_index = 5272

        print("Img path : ", data[1][0][image_index][5][0])

        color_raw = cv2.imread(data[1][0][image_index][5][0], -1)
        depth_raw = cv2.imread(data[1][0][image_index][4][0], -1)
        row = len(color_raw)
        col = len(color_raw[0])

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
                if (items[7].size < 1):
                    continue


                """ 2D bounding box """
                x1 = (int)(items[7][0][0])
                y1 = (int)(items[7][0][1])
                x2 = (int)(items[7][0][2])
                y2 = (int)(items[7][0][3])

                color_raw = cv2.rectangle(color_raw, (x1, y1), (x2, y2), (255,0,0), 2)

                pass
            pass


        # cv2.imshow("color", color_raw)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    else:
        continue
    pass


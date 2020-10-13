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
#
# fig = pyplot.figure()
# ax = Axes3D(fig)

mat_file = io.loadmat('/home/user/SUNRGBD/SUNRGBDMeta.mat')


for range_index in range(4) :

    data = mat_file.popitem()
    if(data[0] == 'SUNRGBDMeta'):
        """
            data[1][0][index] --> index 번째 data. ( 0 <= index <= 10335 )
            data[1][0][index][index2] --> index 번째 data의 index2 번째 data. Matlab에서 column에 해당. 13개.
        """

        for i in range(10335):
            fig = pyplot.figure()
            ax = Axes3D(fig)

            image_index = i

            print("Img path : ", data[1][0][image_index][5][0])

            # color_raw = cv2.imread(data[1][0][image_index][5][0], cv2.COLOR_BGR2RGB)
            color_raw = cv2.imread(data[1][0][image_index][5][0], cv2.COLOR_RGB2BGR)
            depth_raw = cv2.imread(data[1][0][image_index][4][0], -1)

            print("color raw type : ", type(color_raw))

            # print(np.size(color_raw[0]))
            # print(np.ndim(color_raw[0]))
            # print(type(color_raw[0][0]))
            # print(np.ndim(depth_raw))

            """
                uint8 color_raw data to float32 range(0,1)
            """
            # print("type : " , type(color_raw))
            # print("color shape : " , color_raw.shape)
            # print("dtype : ", color_raw.dtype)

            # print("type : " , type(depth_raw))
            # print("shape : " , depth_raw.shape)
            # print("dtype : ", depth_raw.dtype)

            rgb = np.reshape(color_raw, (len(color_raw)*len(color_raw[0]), 3))
            rgb = rgb.astype("float32")
            rgb = rgb / 255
            # cv2.imshow(rgb)
            # print()
            # print(rgb[0:10,0,:])
            # print()
            # print(rgb[0:10,:])
            # print()



            """
                Make 3d point cloud by using depth_raw
            """
            depthInpaint = (depth_raw>>3) | (depth_raw<<(16-3))
            depthInpaint = depthInpaint.astype("float32")
            # depthInpaint = depth_raw/65535*255
            # depthInpaint = depthInpaint.astype("uint8")

            # print("depth shape : " , depthInpaint.shape)

            # transformed = np.dstack([color_raw, depthInpaint])

            # print("transformed shape : " , transformed.shape)

            # print(data[1][0][image_index][5][0])
            # s = data[1][0][image_index][5][0][0:-3]+"tiff"
            # print(s)


            # print(depthInpaint)
            # print(depthInpaint.dtype)

            # cv2.imshow("color", color_raw)
            # cv2.imshow("Red", color_raw[:,:,2])
            # cv2.imshow("Green", color_raw[:,:,1])
            # cv2.imshow("Blue", color_raw[:,:,0])
            # cv2.imshow("Depth", depth_raw)
            # # cv2.imshow("depth transformed", depthInpaint)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            depthInpaint = depthInpaint / 1000
            for row in depthInpaint :
                for ele in row :
                    ele = 8 if ele > 8 else ele
                    pass
                pass

            K = data[1][0][image_index][3]
            cx = K[0][2]
            cy = K[1][2]
            fx = K[0][0]
            fy = K[1][1]

            # print("K : ",K)

            range_x = np.arange(1, len(depth_raw[0])+1)
            range_y = np.arange(1, len(depth_raw)+1)

            x, y = np.meshgrid(range_x, range_y)

            x3 = (x-cx)*depthInpaint*1/fx
            y3 = (y-cy)*depthInpaint*1/fy
            z3 = depthInpaint

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

            print("Rtilt")
            print(Rtilt)
            inv = np.matrix(Rtilt).I
            # inv = np.array.linalg.inv(Rtilt)
            print("inverse")
            print(inv)
            # print("point3d shape : ", point3d.shape)
            # print("rgb shape : ", rgb.shape)


            """
                Random sampling.
                260631 --> 10000.
            """
            sample_size = np.random.randint(len(point3d[0]), size=20000)
            x3 = point3d[0, sample_size]
            y3 = point3d[2, sample_size]
            z3 = point3d[1, sample_size]
            rgb = rgb[sample_size, :]


            """
                Make 3d bounding boxes
            """

            # bounding box size : 64792
            # for i in range(10335):
            #     d = data[1][0][i]
            #     for groundtruth3DBB in d[1]:
            #         for items in groundtruth3DBB:
            #             # print(groundtruth3DBB.size)   : 10
            #             inds = np.argsort(-abs(items[0][:, 0]))
            #             # print("inds : ", inds)
            #             basis = items[0][inds, :]
            #             coeffs = items[1][0, inds]
            #             inds = np.argsort(-abs(basis[1:, 1]))
            #             # print("inds : ", inds)
            #             # print(basis)
            #             # print(coeffs)
            #             pass
            #         pass
            #     pass

            """
                Visualize
            """
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
                    # print(groundtruth3DBB.size)   : 10
                    basis_ori = items[0]

                    # print("basis : \n",basis_ori,"\n")
                    # print("coeffs : \n",items[1][0],"\n")
                    # print("-abs(items[0][:, 0]) : \n", -abs(items[0][:, 0]), "\n")

                    label = items[3][0]

                    if label == "bowl":
                        continue

                    print("label : " , label)
                    inds = np.argsort(-abs(items[0][:, 0]))

                    # print("inds : ",inds,"\n")

                    basis = items[0][inds, :]
                    coeffs = items[1][0, inds]

                    # basis = items[0]
                    # coeffs = items[1][0]

                    # print("basis : \n",basis,"\n")
                    # print("basis : ",basis)
                    # print("coeffs : ",coeffs)

                    inds = np.argsort(-abs(basis[1:, 1]))

                    centroid = items[2]
                    basis = flip_towards_viewer(basis, np.matlib.repmat(centroid, 3, 1))
                    coeffs = abs(coeffs)

                    orientation = items[6][0]

                    for i in range(3):
                        for j in range(3):
                            if basis[i,j] > 1:
                                basis[i,j] = 1

                    print("basis : \n",basis,"\n")

                    print("centroid : \n",centroid,"\n")
                    print("coeffs : \n",coeffs,"\n")

                    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
                    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
                    corners[2, :] = basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
                    corners[3, :] = -basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

                    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
                    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
                    corners[6, :] = basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
                    corners[7, :] = -basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]

                    if label == "bowl":
                        print("corners : ")
                        print(corners)
                    corners += np.matlib.repmat(centroid, 8, 1)
                    #
                    print("corners : ")
                    print(corners)
                    # print("===================================")


                    ax.plot([corners[0, 0], corners[1, 0]], [corners[0, 1], corners[1, 1]],
                            zs=[corners[0, 2], corners[1, 2]], c='r')
                    ax.plot([corners[1, 0], corners[2, 0]], [corners[1, 1], corners[2, 1]],
                            zs=[corners[1, 2], corners[2, 2]], c='r')
                    ax.plot([corners[2, 0], corners[3, 0]], [corners[2, 1], corners[3, 1]],
                            zs=[corners[2, 2], corners[3, 2]], c='r')
                    ax.plot([corners[3, 0], corners[0, 0]], [corners[3, 1], corners[0, 1]],
                            zs=[corners[3, 2], corners[0, 2]], c='r')

                    ax.plot([corners[4, 0], corners[5, 0]], [corners[4, 1], corners[5, 1]],
                            zs=[corners[4, 2], corners[5, 2]], c='r')
                    ax.plot([corners[5, 0], corners[6, 0]], [corners[5, 1], corners[6, 1]],
                            zs=[corners[5, 2], corners[6, 2]], c='r')
                    ax.plot([corners[6, 0], corners[7, 0]], [corners[6, 1], corners[7, 1]],
                            zs=[corners[6, 2], corners[7, 2]], c='r')
                    ax.plot([corners[7, 0], corners[4, 0]], [corners[7, 1], corners[4, 1]],
                            zs=[corners[7, 2], corners[4, 2]], c='r')

                    ax.plot([corners[0, 0], corners[4, 0]], [corners[0, 1], corners[4, 1]],
                            zs=[corners[0, 2], corners[4, 2]], c='r')
                    ax.plot([corners[1, 0], corners[5, 0]], [corners[1, 1], corners[5, 1]],
                            zs=[corners[1, 2], corners[5, 2]], c='r')
                    ax.plot([corners[2, 0], corners[6, 0]], [corners[2, 1], corners[6, 1]],
                            zs=[corners[2, 2], corners[6, 2]], c='r')
                    ax.plot([corners[3, 0], corners[7, 0]], [corners[3, 1], corners[7, 1]],
                            zs=[corners[3, 2], corners[7, 2]], c='r')

                    ax.text3D(corners[0,0], corners[0,1], corners[0,2], label, fontsize=15, color='blue')
                    pass
                pass

            bgr = np.zeros((len(rgb),3))
            bgr[:,0] = rgb[:,2]
            bgr[:,1] = rgb[:,1]
            bgr[:,2] = rgb[:,0]
            ax.scatter(x3, z3, y3, s=1, c=bgr, depthshade=False)
            pyplot.show()

    else:
        continue
    pass


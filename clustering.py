import numpy as np
import cv2
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt


cam_cx = 319.334
cam_cy = 242.507
cam_fx = 611.136
cam_fy = 611.214
K = [[cam_fx, 0, cam_cx],
     [0, cam_fy, cam_cy],
     [0, 0, 1]]
K = np.array(K)
cam_scale = 0.001

xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])

cloud_cluster = o3d.geometry.PointCloud()

color = np.array(Image.open("/home/user/GraspingResearch/img/color_4.png"))
depth = np.array(Image.open("/home/user/GraspingResearch/img/depth_4.png"))
seg = np.array(Image.open("/home/user/GraspingResearch/seg_results/u2-net/Figure_1-3.png"))

# color = np.array(Image.open("/home/user/GraspingResearch/img/color_2.png"))
# depth = np.array(Image.open("/home/user/GraspingResearch/img/depth_2.png"))
# seg = np.array(Image.open("/home/user/GraspingResearch/seg_results/u2-net/Figure_1-1.png"))

seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
# mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
mask = np.logical_and(depth, seg)
# mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
# mask = mask_label * mask_depth
masked_depth = mask * depth * cam_scale

# cv2.imshow("masked_depth", masked_depth)
# cv2.waitKey(0)

pt2 = masked_depth
pt0 = (ymap - cam_cx) * pt2 / cam_fx
pt1 = (xmap - cam_cy) * pt2 / cam_fy

pt0_flatten = pt0.flatten()
pt1_flatten = pt1.flatten()
pt2_flatten = pt2.flatten()
nonzero_pt0_flatten = pt0_flatten[pt0_flatten.nonzero()]
nonzero_pt1_flatten = pt1_flatten[pt1_flatten.nonzero()]
nonzero_pt2_flatten = pt2_flatten[pt2_flatten.nonzero()]

target = np.array([nonzero_pt0_flatten, nonzero_pt1_flatten, nonzero_pt2_flatten]).T
print(target.shape)

cloud_cluster.points = o3d.utility.Vector3dVector(target)

# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
labels = np.array(cloud_cluster.cluster_dbscan(eps=0.01, min_points=10, print_progress=True))
print(labels)
print(labels.shape)
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
# cloud_cluster.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([cloud_cluster])

segmented_points = np.hstack([target, colors])
print(segmented_points.shape)

transposed_points = segmented_points.T
points = transposed_points[:3,:]/transposed_points[2,:]
pred_pixel = K @ points
pred_pixel = pred_pixel.astype(np.int64)
color[pred_pixel[1,:], pred_pixel[0,:]] = segmented_points[:,3:6] * 255
cv2.imshow("color", color)
cv2.waitKey(0)

# for j, indices in enumerate(cluster_indices):
#     # cloudsize = indices
#     print('indices = ' + str(len(indices)))
#     # cloudsize = len(indices)
#     points = np.zeros((len(indices), 3), dtype=np.float32)
#     # points = np.zeros((cloudsize, 3), dtype=np.float32)

#     # for indice in range(len(indices)):
#     for i, indice in enumerate(indices):
#         # print('dataNum = ' + str(i) + ', data point[x y z]: ' + str(cloud_filtered[indice][0]) + ' ' + str(cloud_filtered[indice][1]) + ' ' + str(cloud_filtered[indice][2]))
#         # print('PointCloud representing the Cluster: ' + str(cloud_cluster.size) + " data points.")
#         points[i][0] = cloud_filtered[indice][0]
#         points[i][1] = cloud_filtered[indice][1]
#         points[i][2] = cloud_filtered[indice][2]

#     cloud_cluster.from_array(points)
#     ss = "cloud_cluster_" + str(j) + ".pcd";
    # pcl.save(cloud_cluster, ss)
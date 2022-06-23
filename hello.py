import numpy as np
import cv2
from PIL import Image

# K = np.zeros((3,3))
# K[0,0] = 554.25
# K[1,1] = 554.25
# K[0,2] = 320.5
# K[1,2] = 240.5
# K[2,2] = 1.0
# # print(K)
#
# P = np.zeros((3,1))
# P[0,0] = 182
# P[1,0] = 270
# P[2,0] = 1
# depth = 1.38
# # print(P * depth)
# K = np.linalg.inv(K)
# print(K)
# xyz = (K @ P) * depth
# print(xyz)

# array1 = np.array([[1,2,3], [0,1,4]])
# array2 = np.array([[0,5,1], [2,1,0]])
#
# print(array1)
# print(array2)
# diff = abs(array1 - array2)
# print(abs(array1 - array2))
# print(np.mean(diff))
# print(array1.nonzero())

# depth = cv2.imread('/home/user/6D_pose_estimation_labeled_dataset/depth_0.png', -1)

# depth = np.array(Image.open('/home/user/6D_pose_estimation_labeled_dataset/depth_0.png')).astype(np.float)
# print(depth[depth > 0])
# print(depth.shape)
# depth *= 0.001
# print(depth[320:370, 100:140])
# cv2.imshow("depth", depth)
# cv2.waitKey(0)

# img = cv2.imread('/home/user/6D_pose_estimation_labeled_dataset/color_0.png')
# img2 = cv2.imread('/home/user/6D_pose_estimation_labeled_dataset/color_1.png')
#
# lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
# # l_channel = lab[:,:,0]
#
# lab[:,:,0] = 0
# lab2[:,:,0] = 0
# # lightness_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
# # light_lab = lightness_img
# # light_lab[:,:,0] = l_channel
# # light_img = cv2.cvtColor(light_lab, cv2.COLOR_LAB2BGR)
#
# cv2.imshow("lab", lab)
# # cv2.imshow("lightness_img", np.uint8(lightness_img))
# # cv2.imshow("light_img", np.uint8(light_img))
# cv2.imshow("img", np.uint8(img))
# # print(lightness_img)
# # print(img)
# # diff = lab - lab2
# diff = lab2 - lab
# sum_ = 0
# for i in range(480):
#     for j in range(640):
#         for k in range(1,3):
#             sum_ += diff[i,j,k]
# print(np.sum(diff))
# print(sum_)
# cv2.imshow("diff", diff)
# cv2.waitKey(0)

# img = cv2.imread("/home/user/pose_finder_results/final/scene_1/depth_difference_007_tuna_fish_can_2590.png")
# print(img.shape)
# import numpy as np
# np.random.seed(0)
# print(np.random.random(10))
# print(np.random.random(10))


# densefusion_trans_error_sum = \
# 0.003219432303657416\
# +0.00508775738962155\
# +0.0038128435249786944\
# +0.0024890717892239573\
# +0.0021510998397143316\
# +0.002918100590331272\
# +0.0038255338133276616\
# +0.0017955965340267542\
# +0.007956497239860645\
# +0.006052171763127934\
# +0.003631496410119472\
# +0.004285462492676005\
# +0.007349683151685815\
# +0.003884957056108034\
# +0.005156501035351313\
# +0.00918242488106244\
# +0.006947944790281395\
# +0.0035360890401057226\
# +0.03101574569879635\
# +0.014272976925664036\
# +0.006860667133384274

# print(densefusion_trans_error_sum/21.0)

# my_network_trans_error_sum = \
# +0.0180861995676476\
# +0.026430561837465907\
# +0.017736517831856233\
# +0.017113655634769083\
# +0.01469312948163951\
# +0.015516246613783384\
# +0.015952106902274447\
# +0.01868672800883819\
# +0.013550840303594514\
# +0.017684499654832427\
# +0.017185424590669778\
# +0.019043400717300405\
# +0.019723354516148046\
# +0.01760868394406871\
# +0.014411831277498158\
# +0.016177344165929267\
# +0.015092758514937094\
# +0.017608592004742216\
# +0.034400370183042435\
# +0.02872015755354674\
# +0.013579381293205845

# print(my_network_trans_error_sum/21.0)


# weights = abs(np.random.normal(0.5, 0.5, 100))
# weights = weights / sum(weights)
# # print(weights)
# N = len(weights)
# positions = (np.random.random() + np.arange(N)) / N

# indexes = np.zeros(N, 'i')
# cumulative_sum = np.cumsum(weights)

# print(positions)
# print(indexes)
# print(cumulative_sum)

# import numpy as np

# np.random.seed(76923)

# a = np.random.uniform(1, 2, (2, 2))
# b = np.random.lognormal(3, 1, (2, 2))
# c = np.random.laplace(0, 1, (2, 2))

# print(a)
# print(b)
# print(c)

X = [
    [-1.4, -0.4, -0.4, -0.4, -0.4, -0.4, 0.6, 0.6, 0.6, 1.6],
    [-1.7, -0.7, -0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1.3],
]
X = np.asarray(X)
print(X @ X.T)
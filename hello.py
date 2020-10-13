import numpy as np
import cv2
from scipy import io
print("Hi")

# a = np.zeros((2,3))
# a = [[1,2,3],
#      [4,5,6],
#      [7,8,9],
#      [2,3,4]]
# a = np.array(a)
# print("a")
# print(a)
#
# print("shape : " + str(a.shape))
#
# # 행 바꾸기.
# # a0 = a[:,0].copy()
# # a[:,0] = a[:,1]
# # a[:,1] = a0
# print("a.T")
# print(a.T)
# print("shape : " + str(a.T.shape))
#
# # 모든 열을 2행 값으로 나누기
# a = (a.T / a.T[2,:]).T
# # a = a / a.T[:,2]
#
# print(a)
#
#
# image = cv2.imread("/home/user/0001.png")
# print(str(type(image)))
# print(str(image.shape))
# img = image[:,:,:3]
# cv2.imshow("img", img)
# cv2.waitKey(0)

# K = [[240,320,3],
#      [241,321,6],
#      [239,319,9],
#      [240,321,4],
#      [240,319,4],
#      [241,320,4],
#      [241,319,4],
#      [239,321,4],
#      [239,320,4]]
# K = np.array(K)
# image[K[:,0],K[:,1],0] = 0
# image[K[:,0],K[:,1],1] = 0
# image[K[:,0],K[:,1],2] = 255
#
# cv2.imshow("img", image)
# cv2.waitKey()

# P = [1,2,3,4,5]
# print(P[int(2.3)])

# float_img = np.random.random((40,40))
# # print(float_img)
# uint_img = np.array(float_img*255).astype('uint8')
# # print(uint_img)
# grayImage = cv2.cvtColor(float_img, cv2.COLOR_GRAY2BGR)
# # im = np.array(grayImage * 255, dtype = np.uint8)
# # print(str(type(uint_img)))
# # print(str(uint_img.shape))
# # print(uint_img)
# cv2.imshow("img", grayImage)
# cv2.waitKey(0)

# mat_file = io.loadmat('/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_dataset/data/0006/000001-meta.mat')
# while 1:
#     data = mat_file.popitem()
#     if data is None:
#         print("end of file")
#         break
#     else:
#         if data[0] == 'vertmap':
#             print("size : " + str(data[1].shape))
#             cv2.imshow("img", data[1])
#             cv2.waitKey(0)
#             # print(data[1])

# for i in range(10,20):
#     print(i)

# img = cv2.imread('/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_dataset/data/0001/000001-label.png')
# img = img[:,:,0]
# r = []
# for i in range(480):
#     for j in range(640):
#         if img[i][j] != 0:
#             if img[i][j] in r:
#                 continue
#             else:
#                 r.append(img[i][j])
#
# print(r)

# arr = []
# arr.append(2)
# arr.append(4)
# arr.append(15)
# print(arr)
# print("arr back : " + str(arr[len(arr)-1]))

img = np.zeros((480, 640))
row, col = img.shape

fish_can = cv2.imread("/home/user/label/fish_can.png")
mug = cv2.imread("/home/user/label/mug.png")
spam = cv2.imread("/home/user/label/spam.png")

fish_can = cv2.cvtColor(fish_can, cv2.COLOR_BGR2GRAY)
mug = cv2.cvtColor(mug, cv2.COLOR_BGR2GRAY)
spam = cv2.cvtColor(spam, cv2.COLOR_BGR2GRAY)
# print(img)
max_val = 0
for i in range(row):
    for j in range(col):
        if fish_can[i][j] == 255:
            img[i][j] = 6
        if mug[i][j] == 255:
            img[i][j] = 14
        if spam[i][j] == 255:
            img[i][j] = 9
cv2.imwrite("/home/user/label_float.png", img)

label = img
# label = cv2.imread("/home/user/label.png")
# label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
row,col = label.shape
labels = []
for i in range(row):
    for j in range(col):
        if label[i][j] != 0:
            # img[i][j] = color[label[i][j]-1]
            if label[i][j] not in labels:
                labels.append(label[i][j])

print(labels)

# cv2.imshow("fish_can", fish_can)
# cv2.imshow("mug", mug)
# cv2.imshow("spam", spam)
# cv2.imshow("img", img)
# cv2.waitKey(0)

# img = cv2.imread("/home/user/real_scene_color2.png")
#
# rois = np.zeros((3,4))
#
# rois[0] = [134, 349,  16, 205]
# rois[1] = [116, 232, 209, 346]
# rois[2] = [102, 297, 413, 608]
#
# print(rois)
#
# rois = rois.astype(np.int)
#
# for i in range(3):
#     cv2.rectangle(img, (rois[i,2], rois[i,0]), (rois[i,3], rois[i,1]), (0, 0, 255), 2)
#
# cv2.imshow("img", img)
# cv2.waitKey(0)

import numpy as np
import os
import copy
import random
from PIL import Image
import cv2

dataset_config_dir = '/home/user/python_projects/DenseFusion/datasets/ycb/dataset_config'
dataset_root = "/media/user/ssd_1TB/YCB_dataset"
testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()

# ref_img = cv2.imread("/home/user/Reference_images/mug_1.png")
ref_img = cv2.imread("/home/user/Reference_images/test_screenshot_08.05.2021.png")
# ref_img = cv2.imread("/media/user/ssd_1TB/YCB_dataset/data/0056/000022-color.png")
ref_gray = cv2.cvtColor(ref_img,cv2.COLOR_BGR2GRAY)
cv2.imshow("ref_img", ref_img)
cv2.waitKey(0)
# ref_img = cv2.normalize(ref_img, None, 0, 255, cv2.NORM_MINMAX)
# cv2.imshow("ref_img", ref_img)
# cv2.waitKey(0)
# exit()
noise = np.random.normal(0, 0.1, (480, 640, 3))
img_noise = np.zeros((480, 640, 3), dtype=np.float64)

print(ref_img[150:400, 250:400, :])
img_noise = ref_img + noise
# img_noise[img_noise > 255] = 255
print(img_noise[150:400, 250:400, :])
ref_img = img_noise
# ref_gray = np.add(ref_gray, noise, casting='unsafe')
# print(noise)
# cv2.imshow("img_noise", img_noise)
# cv2.waitKey(0)
# exit()
# ref_gray +=
sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(ref_img,None)

div = 4
for now in range(0, 2949):
    # img = Image.open('{0}/{1}-color.png'.format(dataset_root, testlist[now]))
    key_img = cv2.imread('{0}/{1}-color.png'.format(dataset_root, testlist[now]))
    # for i in range(div):
    #     key_img = key_img_orig[int(i*(480/div)):int((i+1)*(480/div)), int(i*(640/div)):int((i+1)*(640/div)), :]
    #     # cv2.imshow("crop", key_img)
    #     # cv2.waitKey(0)
    #     key_gray= cv2.cvtColor(key_img,cv2.COLOR_BGR2GRAY)
    #
    #     keypoints_2, descriptors_2 = sift.detectAndCompute(key_img,None)
    #
    #     #feature matching
    #     bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    #
    #     matches = bf.match(descriptors_1,descriptors_2)
    #     matches = sorted(matches, key = lambda x:x.distance)
    #
    #     img3 = cv2.drawMatches(ref_img, keypoints_1, key_img, keypoints_2, matches[:10], None, flags=2)
    #
    #     cv2.imshow("matching", img3)
    #     cv2.waitKey(0)

    key_gray = cv2.cvtColor(key_img, cv2.COLOR_BGR2GRAY)

    keypoints_2, descriptors_2 = sift.detectAndCompute(key_img, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(ref_img, keypoints_1, key_img, keypoints_2, matches[:50], None, flags=2)

    cv2.imshow("matching", img3)
    cv2.waitKey(0)
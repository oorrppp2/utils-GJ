import numpy as np
import os
import copy
import random
from PIL import Image
import cv2

from PIL import Image as PIL_Image

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
# ref_img = cv2.imread("/home/user/Reference_images/test_screenshot_08.05.2021.png")
# ref_img = cv2.imread("/media/user/ssd_1TB/YCB_dataset/data/0056/000022-color.png")

# noise = np.random.normal(0, 0.1, (480, 640, 3))
# img_noise = np.zeros((480, 640, 3), dtype=np.float64)
#
# print(ref_img[150:400, 250:400, :])
# img_noise = ref_img + noise
# # img_noise[img_noise > 255] = 255
# print(img_noise[150:400, 250:400, :])
# ref_img = img_noise


sift = cv2.ORB_create()

div = 4
# for now in range(0, 2949):
for now in range(9):   # Scene 10
    for i in range(4):   # Scene 10
        ref_img = cv2.imread("/home/user/Reference_images/005_tomato_soup_can_{0}.png".format(i+2), 1)
        ref_gray = cv2.cvtColor(ref_img,cv2.COLOR_BGR2GRAY)
        keypoints_1, descriptors_1 = sift.detectAndCompute(ref_gray,None)

        key_img = cv2.imread('/home/user/6D_pose_estimation_kist_dataset/color_{0}.png'.format(now))
        # key_img = cv2.imread('{0}/{1}-color.png'.format(dataset_root, testlist[now]))
        key_gray = cv2.cvtColor(key_img, cv2.COLOR_BGR2GRAY)

        keypoints_2, descriptors_2 = sift.detectAndCompute(key_gray, None)

        # # feature matching
        # index_params = dict(algorithm=6,
        #                     table_number=6,  # 12
        #                     key_size=12,  # 20
        #                     multi_probe_level=1)  # 2
        # search_params = dict(checks=50)  # or pass empty dictionary
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        # matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
        # # Need to draw only good matches, so create a mask
        # matchesMask = [[0, 0] for i in range(len(matches))]
        # # ratio test as per Lowe's paper
        # for i, (m, n) in enumerate(matches):
        #     if m.distance < 0.3 * n.distance:
        #         matchesMask[i] = [1, 0]
        # draw_params = dict(matchColor=(0, 255, 0),
        #                    singlePointColor=(255, 0, 0),
        #                    matchesMask=matchesMask,
        #                    flags=0)
        # img3 = cv2.drawMatchesKnn(ref_img, keypoints_1, key_img, keypoints_2, matches, None, **draw_params)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors_1, descriptors_2)# 매칭 결과를 거리기준 오름차순으로 정렬 ---③
        matches = sorted(matches, key=lambda x:x.distance)
        # 모든 매칭점 그리기 ---④
        res1 = cv2.drawMatches(ref_img, keypoints_1, key_img, keypoints_2, matches, None, \
                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        # 매칭점으로 원근 변환 및 영역 표시 ---⑤
        src_pts = np.float32([ keypoints_1[m.queryIdx].pt for m in matches ])
        dst_pts = np.float32([ keypoints_2[m.trainIdx].pt for m in matches ])
        # RANSAC으로 변환 행렬 근사 계산 ---⑥
        mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h,w = ref_img.shape[:2]
        pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
        dst = cv2.perspectiveTransform(pts,mtrx)
        key_img = cv2.polylines(key_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        # 정상치 매칭만 그리기 ---⑦
        matchesMask = mask.ravel().tolist()
        res2 = cv2.drawMatches(ref_img, keypoints_1, key_img, keypoints_2, matches, None, \
                            matchesMask = matchesMask,
                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # 모든 매칭점과 정상치 비율 ---⑧
        accuracy=float(mask.sum()) / mask.size
        print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))

        # 결과 출력
        cv2.imshow('Matching-All', res1)
        cv2.imshow('Matching-Inlier ', res2)

        # img3 = cv2.drawMatches(ref_img, keypoints_1, key_img, keypoints_2, matches[:50], None, flags=2)

        # cv2.imshow("matching", img3)
        cv2.waitKey(0)
import numpy as np
import os
import copy
import random
from PIL import Image
import cv2

import pyrealsense2 as rs

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

sift = cv2.SIFT_create()
ref_img = cv2.imread("/home/user/Reference_images/005_tomato_soup_can_{0}.png".format(2), 1)
ref_gray = cv2.cvtColor(ref_img,cv2.COLOR_BGR2GRAY)
keypoints_1, descriptors_1 = sift.detectAndCompute(ref_gray,None)


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))


config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

align_to= rs.stream.color
align= rs.align(align_to)

# Start streaming
pipeline.start(config)
sq = 0

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames= align.process(frames)
        aligned_depth_frame= aligned_frames.get_depth_frame()
        color_frame= aligned_frames.get_color_frame()

        # processed = d.process(aligned_depth_frame)
        prof = aligned_depth_frame.get_profile()
        video_prof = prof.as_video_stream_profile()
        intr = video_prof.get_intrinsics()
        sq += 1
        if not color_frame:
            continue

        depth_image= np.asanyarray(aligned_depth_frame.get_data())
        key_img = np.asanyarray(color_frame.get_data())

        key_gray = cv2.cvtColor(key_img, cv2.COLOR_BGR2GRAY)

        keypoints_2, descriptors_2 = sift.detectAndCompute(key_img, None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        print(descriptors_1.shape)
        print(descriptors_2.shape)
        matches = matcher.match(descriptors_1, descriptors_2)  # 매칭 결과를 거리기준 오름차순으로 정렬 ---③
        matches = sorted(matches, key=lambda x: x.distance)
        # 모든 매칭점 그리기 ---④
        res1 = cv2.drawMatches(ref_img, keypoints_1, key_img, keypoints_2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        # 매칭점으로 원근 변환 및 영역 표시 ---⑤
        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches])
        # RANSAC으로 변환 행렬 근사 계산 ---⑥
        mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = ref_img.shape[:2]
        pts = np.float32([[[0, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w - 1, 0]]])
        dst = cv2.perspectiveTransform(pts, mtrx)
        key_img = cv2.polylines(key_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # 정상치 매칭만 그리기 ---⑦
        matchesMask = mask.ravel().tolist()
        res2 = cv2.drawMatches(ref_img, keypoints_1, key_img, keypoints_2, matches, None,
                               matchesMask=matchesMask,
                               flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # 모든 매칭점과 정상치 비율 ---⑧
        accuracy = float(mask.sum()) / mask.size
        print("accuracy: %d/%d(%.2f%%)" % (mask.sum(), mask.size, accuracy))

        # 결과 출력
        cv2.imshow('Matching-All', res1)
        cv2.imshow('Matching-Inlier ', res2)
        cv2.waitKey(1)

        # img3 = cv2.drawMatches(ref_img, keypoints_1, key_img, keypoints_2, matches[:50], None, flags=2)

        # cv2.imshow("matching", img3)
finally:
    # Stop streaming
    pipeline.stop()
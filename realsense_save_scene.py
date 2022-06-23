## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# print(device_product_line)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

align_to= rs.stream.color
align= rs.align(align_to)

# Start streaming
pipeline.start(config)
sq = 0

# print(config)

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
        print(intr)
        sq += 1
        if not color_frame:
            continue

        depth_image= np.asanyarray(aligned_depth_frame.get_data())
        color_image= np.asanyarray(color_frame.get_data())

        # Show images
        key = cv2.waitKey(1)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        if key == ord('s'):
            # cv2.imwrite("/home/user/6D_pose_estimation_kist_dataset/color_"+str(sq)+".png", color_image)
            # cv2.imwrite("/home/user/6D_pose_estimation_kist_dataset/depth_"+str(sq)+".png", depth_image)
            # cv2.imwrite("/home/user/camera_calibration/color_"+str(sq)+".png", color_image)
            # cv2.imwrite("/home/user/ycb_test_images/color_"+str(sq)+".png", color_image)
            # cv2.imwrite("/home/user/ycb_test_images/depth_"+str(sq)+".png", depth_image)
            cv2.imwrite("/home/user/GraspingResearch/img/color_"+str(sq)+".png", color_image)
            cv2.imwrite("/home/user/GraspingResearch/img/depth_"+str(sq)+".png", depth_image)
            print("Saved reference image!")
        elif key == ord('q'):
            break
        cv2.imshow('RealSense', color_image)
        # cv2.imshow('RealSense_depth', depth_image)

finally:

    # Stop streaming
    pipeline.stop()
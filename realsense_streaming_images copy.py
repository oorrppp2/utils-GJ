# ## License: Apache 2.0. See LICENSE file in root directory.
# ## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

# ###############################################
# ##      Open CV and Numpy integration        ##
# ###############################################

# import pyrealsense2 as rs
# import numpy as np
# import cv2
# from pyrealsense2 import disparity_frame

# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()

# # Get device product line for setting a supporting resolution
# pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# pipeline_profile = config.resolve(pipeline_wrapper)
# device = pipeline_profile.get_device()
# device_product_line = str(device.get_info(rs.camera_info.product_line))


# # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y16, 30)
# config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y16, 30)

# # align_to= rs.stream.color
# # align= rs.align(align_to)

# # config.disable_stream(rs.stream.depth)

# # Start streaming
# pipeline.start(config)
# sq = 0

# # print(config)

# try:
#     while True:

#         # Wait for a coherent pair of frames: depth and color
#         frames = pipeline.wait_for_frames()
#         aligned_frames= align.process(frames)
#         aligned_depth_frame= aligned_frames.get_depth_frame()
#         color_frame= aligned_frames.get_color_frame()

#         # processed = d.process(aligned_depth_frame)
#         prof = aligned_depth_frame.get_profile()
#         video_prof = prof.as_video_stream_profile()
#         intr = video_prof.get_intrinsics()
#         print(frames.get_baseline())
#         print(intr)
#         sq += 1
#         if not color_frame:
#             continue

#         depth_image= np.asanyarray(aligned_depth_frame.get_data())
#         color_image= np.asanyarray(color_frame.get_data())

#         # Show images
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
#         cv2.imshow('RealSense', color_image)
#         # cv2.imshow('RealSense_depth', depth_image)

# finally:

#     # Stop streaming
#     pipeline.stop()

import pyrealsense2 as rs
import numpy as np
import cv2

xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 323.467
cam_cy = 241.674
cam_fx = 383.472
cam_fy = 383.472
cam_scale = 0.001
K = np.array([[cam_fx , 0 , cam_cx],
     [0, cam_fy, cam_cy],
     [0, 0, 1]])

color_to_infra1 = 0.015
color_to_infra2 = 0.065

world_to_infra1_rotation = np.array([[1.000,-0.005, 0.004],
                                    [0.005, 1.000, 0.000],
                                    [-0.004, 0.000, 1.000]])
world_to_infra2_rotation = np.array([[1.000,0.000, 0.001],
                                    [0.000, 1.000, 0.000],
                                    [-0.001, 0.000, 1.000]])
infra_1_intrinsic = np.array([[0.500,0.799, 0.502],
                            [0.503, -0.056, 0.064],
                            [-0.001, -0.001, -0.021]])
infra_2_intrinsic = np.array([[0.498,0.797, 0.504],
                            [0.506, -0.058, 0.066],
                            [0.000, -0.001, -0.021]])
world_to_infra1 = infra_1_intrinsic
world_to_infra2 = world_to_infra2_rotation @ infra_2_intrinsic

points = rs.points()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

color_u = (xmap - cam_cx)/cam_fx - 0.015
color_v = (ymap - cam_cy)/cam_fy
infra1_x = world_to_infra1_rotation[0,0] * color_u + world_to_infra1_rotation[0,1] * color_v +  world_to_infra1_rotation[0,2]
infra1_y = world_to_infra1[1,0] * color_u + world_to_infra1[1,1] * color_v +  world_to_infra1[1,2]
infra1_scale = world_to_infra1[2,0] * color_u + world_to_infra1[2,1] * color_v +  world_to_infra1[2,2]
infra1_x /= infra1_scale 
infra1_y /= infra1_scale 

# print(infra1_x)
# print(infra1_y)
# print(len(infra1_x[infra1_x >1]))

empty_image = np.zeros((480, 640, 3))
# print(infra1_scale)
# exit(0)
try:
    while True:
        frames = pipeline.wait_for_frames()
        ir1_frame = frames.get_infrared_frame(1)
        ir2_frame = frames.get_infrared_frame(2)

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not ir1_frame:
            continue
        image1 = np.asanyarray(ir1_frame.get_data())
        image2 = np.asanyarray(ir2_frame.get_data())
        color = np.asanyarray(color_frame.get_data())

        for i in range(480):
            for j in range(640):
                if infra1_y[i,j] > 0 and infra1_x[i,j] > 0:
                    # print(infra1_y[i,j], ", ", infra1_x[i,j])
                    empty_image[i,j,:] = color[int(infra1_y[i,j]), int(infra1_x[i,j]),:]
        cv2.imshow('empty_image', empty_image)
        cv2.imshow('IR1 Example', image1)
        cv2.imshow('IR2 Example', image2)
        cv2.imshow('color', color)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
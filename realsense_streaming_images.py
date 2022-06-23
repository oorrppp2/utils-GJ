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

points = rs.points()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

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
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#####################################################
## librealsense tutorial #1 - Accessing depth data ##
#####################################################

# First import the library
import pyrealsense2 as rs
import cv2
import numpy as np

try:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)
    pipeline.start(config)

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        r, g, b = cv2.split(color_image)
        color_image = cv2.merge((b, g, r))

        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())

        cv2.imshow('color_image', color_image)
        cv2.imshow('aligned_depth_image', aligned_depth_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    exit(0)
except Exception as e:
    print(e)
    pass
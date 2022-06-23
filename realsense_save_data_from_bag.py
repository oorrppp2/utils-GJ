import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
# dir = "20200727_195335.bag"

def main():
    # if not os.path.exists(args.directory):
    #     os.mkdir(args.directory)
    try:
        # for s in range(15,16):
        file_name = "20211005_174637"
        read_path ="/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/ZiPSA_video_bag/" + file_name + ".bag"
        save_path = "/home/user/ZiPSA_6D_pose_estimation/"
        begin = time.time()

        config = rs.config()
        config.enable_stream(rs.stream.color)
        config.enable_stream(rs.stream.depth)
        pipeline = rs.pipeline()
        rs.config.enable_device_from_file(config, read_path)
        profile = pipeline.start(config)

        align_to = rs.stream.color
        align = rs.align(align_to)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("scale : " + str(depth_scale))

        profile = pipeline.get_active_profile()

        color_stream = profile.get_stream(rs.stream.color)
        color_profile = rs.video_stream_profile(color_stream)
        color_intrinsics = color_profile.get_intrinsics()

        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()

        print("*** color intrinsics ***")
        print(color_intrinsics)

        print("*** depth intrinsics ***")
        print(depth_intrinsics)

        save_images = []
        save_depth = []

        while True:
            if time.time() - begin > 6:
                break
            time.sleep(0.02)

            frames = pipeline.wait_for_frames()
            # align the deph to color frame
            aligned_frames = align.process(frames)
            # get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
            # scaled_depth_image = depth_image * depth_scale

            # convert color image to BGR for OpenCV
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            r, g, b = cv2.split(color_image)
            color_image = cv2.merge((b, g, r))

            if time.time() - begin > 3:
                cv2.imshow("color", color_image)
                cv2.imshow("aligned_depth_image", aligned_depth_image)
                cv2.waitKey(1)
                # save_images.append(color_image.copy())
                # save_depth.append(aligned_depth_image.copy())


        # for i in range(len(save_images)):
        #     cv2.imwrite(save_path+"depth_" + str(i) + ".png", save_depth[i])
        #     cv2.imwrite(save_path+"color_" + str(i) + ".png", save_images[i])
    finally:
        pass


if __name__ == "__main__":

    main()

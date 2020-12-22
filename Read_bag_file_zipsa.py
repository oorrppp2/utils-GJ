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
        for s in range(15,16):
            read_path ="/home/user/python_projects/utils-GJ/realsense_bag/kist_scene/"+str(s)+".bag"
            save_path = "/home/user/python_projects/DenseFusion_yolact_base/living_lab_video/"+str(s)+"/"
            begin = time.time()

            index = 0
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

            while True:
                if time.time() - begin > 22:
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
                # r, g, b = cv2.split(color_image)
                # color_image = cv2.merge((b, g, r))

                # cv2.imshow("color", color_image)
                # cv2.imshow("aligned_depth_image", aligned_depth_image)
                # cv2.waitKey(0)
                print("index : " + str(index))

                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                cv2.imwrite(save_path+"depth_image" + str(index) + ".png", aligned_depth_image)
                cv2.imwrite(save_path+"color_image" + str(index) + ".png", color_image)
                # cv2.imwrite("/home/user/kist_scene/scene" + str(i) + "/depth_image" + str(index) + ".png", aligned_depth_image)
                # cv2.imwrite("/home/user/kist_scene/scene" + str(i) + "/color_image" + str(index) + ".png", color_image)
                index += 1
    finally:
        pass


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--directory", type=str, help="Path to save the images")
    # parser.add_argument("-i", "--input", type=str, help="Bag file to read")
    # args = parser.parse_args()

    main()
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
        # file_name = "20211005_175517"
        file_name = "6"
        
        read_path ="/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/IROS22_Video/" + file_name + ".bag"
        save_path = "/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/IROS22_Video/"      
        # read_path ="/home/user/Documents/" + file_name + ".bag"
        # save_path = "/home/user/Documents/"
        begin = time.time()

        index = 0
        config = rs.config()
        config.enable_stream(rs.stream.color)
        pipeline = rs.pipeline()
        rs.config.enable_device_from_file(config, read_path)
        profile = pipeline.start(config)

        profile = pipeline.get_active_profile()

        color_stream = profile.get_stream(rs.stream.color)
        color_profile = rs.video_stream_profile(color_stream)
        save_images = []

        while True:
            if time.time() - begin > 169:
                break
            # time.sleep(0.02)
            frames = pipeline.wait_for_frames()
            # convert color image to BGR for OpenCV
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            r, g, b = cv2.split(color_image)
            color_image = cv2.merge((b, g, r))

            save_images.append(color_image)

        height, width, _ = save_images[0].shape
        size = (width,height)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        video = cv2.VideoWriter(save_path+file_name+'.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
        for index, img in enumerate(save_images):
            video.write(img)
            print("index : " + str(index))
        video.release()
    finally:
        pass


if __name__ == "__main__":

    main()
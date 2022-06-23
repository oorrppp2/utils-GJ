import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
# dir = "20200727_195335.bag"

def main():
    save_path = "/home/user/python_projects/6D_pose_estimation_particle_filter/results/video/"
    model_name = '011_banana'

    height, width = (480, 640)
    size = (width,height)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    video = cv2.VideoWriter(save_path+model_name+'.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 7.954202488, size)

    for index in range(30):
        frame_time = time.time()
        img = cv2.imread("/home/user/python_projects/6D_pose_estimation_particle_filter/results/images/{0}_{1}.png".format(model_name, str(index)))
        # time.sleep(0.093976784)

        # while time.time() - frame_time < 0.093976784:
        video.write(img)
        print("index : " + str(index))
    video.release()


if __name__ == "__main__":

    main()
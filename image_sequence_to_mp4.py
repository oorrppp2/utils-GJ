import numpy as np
import cv2
import os
import time
# dir = "20200727_195335.bag"

root_dir = '/home/user/ZiPSA_6D_pose_estimation/results_revise_3_image/'
def main():
    try:
        file_name = "results_image_v3"
        save_images = []
        for now in range(84):
            color_image = cv2.imread(root_dir+str(now)+".png")
            # r, g, b = cv2.split(color_image)
            # color_image = cv2.merge((b, g, r))

            save_images.append(color_image)

        height, width, _ = save_images[0].shape
        size = (width,height)
        video = cv2.VideoWriter(root_dir+file_name+'.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
        for index, img in enumerate(save_images):
            video.write(img)
            print("index : " + str(index))
        video.release()
    finally:
        pass


if __name__ == "__main__":

    main()
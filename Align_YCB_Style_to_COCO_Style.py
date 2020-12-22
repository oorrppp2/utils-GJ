from PIL import Image 					# (pip install Pillow)
import numpy as np                                 	# (pip install numpy)
import os
import cv2
import json
import shutil


def move_syn_images(directory):
    for root, dirs, files in os.walk(os.path.abspath(directory)):
        if "data_syn" in root:
            root_num = root[-1]
            print(root)
            for i in range(int(root_num)*10000, int(root_num)*10000+10000, 4):
                path_head = str(i).zfill(6)
                root += '/'
                color_path = root + path_head + "-color.png"
                label_path = root + path_head + "-label.png"
                meta_path = root + path_head + "-meta.mat"
                shutil.copy(color_path, '/media/user/ssd_1TB/YCB_dataset_coco_style/train/syn')
                shutil.copy(label_path, '/media/user/ssd_1TB/YCB_dataset_coco_style/train/syn')
                shutil.copy(meta_path, '/media/user/ssd_1TB/YCB_dataset_coco_style/train/syn')


def move_data_images(directory):
    move = []
    all_num_files = 0

    num_files = 0
    path_head = directory
    for i in range(1):
        folder_num = str(i).zfill(4)
        path = path_head + folder_num
        print(path)
        for root, dirs, files in os.walk(os.path.abspath(path)):
            root += '/'
            for file in files:
                if int(file[:6]) % 4 != 0:
                    continue
                all_num_files += 1
                if "label" in file:
                    num_files += 1
                    file_id = int(file[:6]) + i*10000
                    save_file_name = str(file_id).zfill(6)
                    save_file_dir = "/media/user/ssd_1TB/YCB_dataset_coco_style/train/"
                    save_file_name = save_file_dir + save_file_name + file[6:]

                    # print(root + file)
                    # print(save_file_name)
                    shutil.copy(root+file, save_file_dir)
                    os.rename(save_file_dir+file, save_file_name)

                if "color" in file:
                    num_files += 1
                    file_id = int(file[:6]) + i*10000
                    save_file_name = str(file_id).zfill(6)
                    save_file_dir = "/media/user/ssd_1TB/YCB_dataset_coco_style/train/"
                    save_file_name = save_file_dir + save_file_name + file[6:]

                    # print(root + file)
                    # print(save_file_name)
                    shutil.copy(root+file, save_file_dir)
                    os.rename(save_file_dir+file, save_file_name)

                if "box" in file:
                    num_files += 1
                    file_id = int(file[:6]) + i*10000
                    save_file_name = str(file_id).zfill(6)
                    save_file_dir = "/media/user/ssd_1TB/YCB_dataset_coco_style/train/"
                    save_file_name = save_file_dir + save_file_name + file[6:]

                    # print(root + file)
                    # print(save_file_name)
                    shutil.copy(root+file, save_file_dir)
                    os.rename(save_file_dir+file, save_file_name)
        pass
    # print("size : " + str(len(moves)))
    print("files : " + str(num_files))
    print("all_files : " + str(all_num_files))

if __name__ == '__main__':
    # data_syn
    # move_syn_images("/media/user/ssd_1TB/YCB_dataset/")
    move_data_images("/media/user/ssd_1TB/YCB_dataset/data/")

    # os.remove("/media/user/ssd_1TB/YCB_dataset_coco_style/train/411500-label.png")
    # os.remove("/media/user/ssd_1TB/YCB_dataset_coco_style/train/411500-color.png")

    #remove
    file_num = 0
    for root, dirs, files in os.walk("/media/user/ssd_1TB/YCB_dataset_coco_style/train"):
        root += '/'
        for file in files:
            # if "depth" in file:
                # os.remove(root + file)
            file_num += 1
    print("check files : " + str(file_num))

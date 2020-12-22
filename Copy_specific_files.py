from PIL import Image 					# (pip install Pillow)
import numpy as np                                 	# (pip install numpy)
import os
import cv2
import json
import shutil

dataset_root = '/media/user/ssd_1TB/YCB_dataset/'
ycb_toolbox_dir = '/home/user/python_projects/DenseFusion/YCB_Video_toolbox'
dataset_config_dir = '/home/user/python_projects/DenseFusion/datasets/ycb/dataset_config'
save_root = '/media/user/ssd_1TB/YCB_dataset_meta/'
testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line+"-meta.mat")
input_file.close()
# print(len(testlist))
# print(testlist)

for path in testlist:
    from_path = dataset_root + path
    print(from_path)
    save_path = save_root + path[:9]
    print(save_path)
    shutil.copy(from_path, save_path)

# for i in range(48,60):
#     os.mkdir(save_root+'%04d' % i)

# def move_syn_images(directory):
#     for now in range(0, 2949):
#         meta_path = '{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now)
#
#     for root, dirs, files in os.walk(os.path.abspath(directory)):
#         if "data_syn" in root:
#             root_num = root[-1]
#             print(root)
#             for i in range(int(root_num)*10000, int(root_num)*10000+10000, 4):
#                 path_head = str(i).zfill(6)
#                 root += '/'
#                 color_path = root + path_head + "-color.png"
#                 label_path = root + path_head + "-label.png"
#                 meta_path = root + path_head + "-meta.mat"
#                 shutil.copy(color_path, '/media/user/ssd_1TB/YCB_dataset_coco_style/train/syn')
#                 shutil.copy(label_path, '/media/user/ssd_1TB/YCB_dataset_coco_style/train/syn')
#                 shutil.copy(meta_path, '/media/user/ssd_1TB/YCB_dataset_coco_style/train/syn')

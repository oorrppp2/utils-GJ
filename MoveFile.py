import shutil
import os

# color depth label meta
src = '/media/user/ssd_1TB/YCB_dataset_coco_style/train'
dst = '/home/user/python_projects/yolact/data/ycb_latest/train/img'

cnt = 0
for root, dirs, files in os.walk(src):
    root += '/'
    for file in files:
        pass
        if "color" in file:
            # print(file)
            # cnt += 1
            shutil.copy(root+file, dst)

# print(cnt)
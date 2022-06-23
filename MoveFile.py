import shutil
import os

# color depth label meta
src = '/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_dataset/data_syn'
dst = '/home/user/python_projects/yolact/data/ycb_latest/train/img'

cnt = 0
for i in range(8):
    for root, dirs, files in os.walk(src+str(i)):
        # root += '/'
        print(len(files))
    # for file in files:
    #     # pass
    #     if "color" in file:
    #         print(file)
    #         # cnt += 1
    #         # shutil.copy(root+file, dst)

# print(cnt)
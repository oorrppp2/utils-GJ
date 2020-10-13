import shutil

# color depth label meta
src = '/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_dataset/data_syn/'
dst = '/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_dataset/data_syn2/'
for i in range(20000,30000):
    color = ('%06d-' % i) + 'color.png'
    depth = ('%06d-' % i) + 'depth.png'
    label = ('%06d-' % i) + 'label.png'
    meta = ('%06d-' % i) + 'meta.mat'
    # print(color)
    # print(depth)
    # print(label)
    # print(meta)
    shutil.move(src+color, dst+color)
    shutil.move(src+depth, dst+depth)
    shutil.move(src+label, dst+label)
    shutil.move(src+meta, dst+meta)
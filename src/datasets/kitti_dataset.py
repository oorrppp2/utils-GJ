from __future__ import division
import os
import numpy as np
import cv2
import torch.utils.data as torch_data
import src.utils.kitti_utils as kitti_utils
import src.utils.calibration as calibration

class KittiDataset(torch_data.Dataset):

    def __init__(self, root_dir, split='train', fov=True):
        self.split = split
        self.fov = fov

        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir, 'KITTI', 'object', 'testing' if is_test else 'training')

        split_dir = os.path.join('data', 'KITTI', 'ImageSets', split+'.txt')
        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_samples = self.image_idx_list.__len__()

        if fov:
            self.lidar_path = os.path.join(self.imageset_dir, "velodyne_fov")
        else:
            self.lidar_path = os.path.join(self.imageset_dir, "velodyne")

        self.image_path = os.path.join(self.imageset_dir, "image_2")
        self.calib_path = os.path.join(self.imageset_dir, "calib")
        self.label_path = os.path.join(self.imageset_dir, "label_2")

    def get_image(self, idx):
        img_file = os.path.join(self.image_path, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file) # (H, W, C) -> (H, W, 3) OpenCV reads in BGR mode

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_path, '%06d.png' % idx)
        assert  os.path.exists(img_file)
        img = cv2.imread(img_file)
        width, height, channel = img.shape
        return width, height, channel

    def get_lidar(self, idx):
        if self.fov:
            lidar_file = os.path.join(self.lidar_path, '%06d.npy' % idx)
        else:
            lidar_file = os.path.join(self.lidar_path, '%06d.bin' % idx)

        assert os.path.exists(lidar_file)

        if self.fov:
            return np.load(lidar_file)
        else:
            return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_path, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_path, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    def __len__(self):
        raise NotImplemented

    def __getitem__(self, item):
        raise NotImplemented
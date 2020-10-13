
import numpy as np
from src.datasets.kitti_dataset import KittiDataset
import src.utils.kitti_utils as kitti_utils
import src.utils.kitti_aug_utils as augUtils
import src.utils.utils as utils
import src.config as cnf

class KittiYOLODataset(KittiDataset):

    def __init__(self, root_dir, split='train', mode ='TRAIN', data_aug=True, fov=True):
        super().__init__(root_dir=root_dir, split=split, fov=fov)

        self.split = split
        self.data_aug = data_aug
        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode

        self.sample_id_list = []

        if mode == 'TRAIN':
            self.preprocess_yolo_training_data()
        else:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]

        print('Load %s samples from %s' % (mode, self.imageset_dir))
        print('Done: total %s samples %d' % (mode, len(self.sample_id_list)))

    def preprocess_yolo_training_data(self):
        """
        Discard samples which don't have current training class objects, which will not be used for training.
        Valid sample_id is stored in self.sample_id_list
        """
        for idx in range(0, self.num_samples):
            sample_id = int(self.image_idx_list[idx])
            objects = self.get_label(sample_id)
            labels, noObjectLabels = kitti_utils.read_labels_for_bevbox(objects)
            if not noObjectLabels:
                labels[:, 1:] = augUtils.camera_to_lidar_box(labels[:, 1:])  # convert rect cam to velo cord

            valid_list = []
            for i in range(labels.shape[0]):
                if int(labels[i, 0]) in cnf.CLASS_NAME_TO_ID.values():
                    if self.check_pc_range(labels[i, 1:4]) is True:
                        valid_list.append(labels[i,0])

            if len(valid_list):
                self.sample_id_list.append(sample_id)

    def check_pc_range(self, xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range = [cnf.boundary["minX"], cnf.boundary["maxX"]]
        y_range = [cnf.boundary["minY"], cnf.boundary["maxY"]]
        z_range = [cnf.boundary["minZ"], cnf.boundary["maxZ"]]

        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    def __getitem__(self, index):
        
        sample_id = int(self.sample_id_list[index])
        #img_2d = self.get_image(sample_id)

        if self.mode in ['TRAIN', 'EVAL']:
            lidarData = self.get_lidar(sample_id)    
            objects = self.get_label(sample_id)

            labels, noObjectLabels = kitti_utils.read_labels_for_bevbox(objects)
    
            if not noObjectLabels:
                labels[:, 1:] = augUtils.camera_to_lidar_box(labels[:, 1:], )  # convert rect cam to velo cord

            if self.data_aug and self.mode == 'TRAIN':
                lidarData, labels[:, 1:] = augUtils.voxelNetAugScheme(lidarData, labels[:, 1:], True)

            b = utils.removePoints(lidarData, cnf.boundary)
            rgb_map = utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
            target = kitti_utils.build_yolo_target(labels)
            return rgb_map, target

        else:
            lidarData = self.get_lidar(sample_id)
            b = utils.removePoints(lidarData, kitti_utils.bc)
            rgb_map = utils.makeBVFeature(b, cnf.DISCRETIZATION)
            return rgb_map

    def __len__(self):
        return len(self.sample_id_list)
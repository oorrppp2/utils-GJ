import numpy as np

models = [
        "002_master_chef_can",      # 1
        "003_cracker_box",          # 2
        "004_sugar_box",            # 3
        "005_tomato_soup_can",      # 4
        "006_mustard_bottle",       # 5
        "007_tuna_fish_can",        # 6
        "008_pudding_box",          # 7
        "009_gelatin_box",          # 8
        "010_potted_meat_can",      # 9
        "011_banana",               # 10
        "019_pitcher_base",         # 11
        "021_bleach_cleanser",      # 12
        "024_bowl",                 # 13
        "025_mug",                  # 14
        "035_power_drill",          # 15
        "036_wood_block",           # 16
        "037_scissors",             # 17
        "040_large_marker",         # 18
        "051_large_clamp",          # 19
        "052_extra_large_clamp",    # 20
        "061_foam_brick"            # 21
]

# import torch

# # p1 = torch.FloatTensor([5,5,4])
# # p2 = torch.FloatTensor([10,10,6])
# # p3 = torch.FloatTensor([5,5,5])
# # p4 = torch.FloatTensor([10,10,3])
# # p1 = torch.FloatTensor([-0.7904,  20.1214, 1.6172])
# # p2 = torch.FloatTensor([0.0029, 0.0207, 1.5271])
# # p3 = torch.FloatTensor([-0.6745, -0.2950, -1.2946])
# # p4 = torch.FloatTensor([0.0072, 0.0207, 1.5281])
# p1 = torch.FloatTensor([1,1,1])
# p2 = torch.FloatTensor([2,2,2])
# p3 = torch.FloatTensor([2,-2,0])
# p4 = torch.FloatTensor([2,1,2])

# cp1 = torch.cross((p2-p1),(p4-p3))

# constant1_1 = torch.dot(cp1, p1)
# constant1_2 = torch.dot(cp1, p2)
# print(constant1_1)
# print(constant1_2)

# cp2 = torch.cross((p2-p1), cp1)
# constant2_1 = torch.dot(cp2, p1)
# constant2_2 = torch.dot(cp2, p2)
# print(constant2_1)
# print(constant2_2)

# t = (torch.dot(cp2, p3) - constant2_1) / (-torch.dot(cp2, (p4-p3)))
# print(t)

# intersection = p3 + (p4-p3)*t
# print(intersection)

# # print(np.sum(range(1000)))

# import os.path
# model = "003_cracker_box"
# save_path = '/home/user/pose_finder_results/ycb_pf_cuda_v1/'
# target_str = "1 2 3 \n"
# for model in models:
#     write_str = ""
#     if os.path.isfile(save_path + "result_{0}.txt".format(model)):
#         rf = open(save_path + "result_{0}.txt".format(model))
#         lines = rf.readlines()
#         for line in lines:
#             write_str += line
#         write_str += target_str
#         rf.close()
#     else:
#         write_str += target_str

#     f = open(save_path + "result_{0}.txt".format(model), mode='wt')
#     f.write(write_str)
#     f.close()
    
# for now in range(63,80):
#     print(os.path.isfile("/home/user/pose_finder_results/ycb_pf_results_fast_v3/"+model+"_"+str(now)+".npy"))

# labels = [2,5,7,8,9,22,33,11]

# # np.save("/home/user/pose_finder_results/ycb_pf_results_fast_v3/"+model+"_"+str(now), pose)

# for i in range(int(len(labels) / 5 ) + 1):
#     for j, itemid in enumerate(labels[i*5:(i+1)*5]):
#         print("itemid : ", itemid)
#         print("index : ", (i*5) + j)
#         # print(len(labels[i*5:(i+1)*5]))
#         # print(labels[i*5:(i+1)*5])

# sum = 0
# for i in range(640):
#     sum += i
# print(sum)\


import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torch.nn.functional as F

class PoseNet(nn.Module):
    def __init__(self, num_points):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        # self.feat_input = PoseNetFeat(num_points)
        # self.feat_pcd = PoseNetFeat(num_points)
        
        self.conv1_t = torch.nn.Conv1d(3,6,1)
        # self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        # self.conv1_t = torch.nn.Conv1d(704, 256, 1)
        # self.conv2_t = torch.nn.Conv1d(256, 128, 1)
        # self.conv3_t = torch.nn.Conv1d(128, 3, 1) #translation

        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(128)

    # def forward(self, img, x, choose, obj):
    def forward(self, x):
        # ap_point = self.feat_input(x)

        tx = F.relu(self.conv1_t(x))
        # tx = F.relu(self.bn1(self.conv1_t(tx)))
        # tx = F.relu(self.bn2(self.conv2_t(tx)))
        # tx = self.conv3_t(tx)

        # out_tx = tx.contiguous().transpose(2, 1).contiguous()

        return tx


if __name__ == '__main__':
    net = PoseNet(5)
    pc = torch.rand((1,3, 5))
    print(pc)
    val = net(pc)
    print(val.shape)
    print(net.conv1_t)
    # print(net.parameters())
    print(val)

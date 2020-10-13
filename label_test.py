import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import matplotlib.pyplot as plt
import cv2

label_in = np.array(Image.open('/home/user/PycharmProjects/DenseFusion/datasets/linemod/Linemod_preprocessed/data/04/mask/0000.png'))
# row, col, c = label_in.shape
# print("row : " + str(row) + " / col : " + str(col))
label_in = label_in[:,:,0]
seg_list = [0, 21, 43, 106, 128, 170, 191, 213, 234, 255]
label = np.zeros((len(seg_list), label_in.shape[0], label_in.shape[1]))
print("label shape : " + str(label.shape))
for j in range(len(seg_list)):
    label[j, :] = label_in == seg_list[j]
# label = np.argmax(label, axis=0)
for i in range(480):
    for j in range(640):
        if label[9,i,j] != 0:
            print(label[9,i,j])
# print(label[9].shape)
# print(label[9,200,200])
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
#import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from tqdm import tqdm
import shutil
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pdb
import copy

criterion = nn.CrossEntropyLoss()
print("Simple torch")
input_numpy = np.zeros((3,5))
input_numpy[0,:] = 1
input_numpy[1,:] = 2
input_numpy[2,:] = 2
input = torch.from_numpy(input_numpy.astype(np.float32))
target_numpy = np.zeros((2,3))
target_numpy[0] = 2
target_numpy[1] = 0
target_numpy = np.argmax(target_numpy, axis = 0)

target = torch.from_numpy(target_numpy.astype(int))
print("input : " + str(input))
print("input shape : " + str(input.shape))
print("target : " + str(target))
print("target shape: " + str(target.shape))
loss = criterion(input,target)
print("loss : " + str(loss))
# loss.backward()
import numpy as np
import cv2

import sys

img = cv2.imread('/home/user/6D_pose_estimation_kist_dataset/color_0.png')
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
# Step 1
# rect = (50,50,450,290)
# cv2.grabCut(img,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
# Step 2
# newmask = cv2.imread('./data/newmask2.png',0)
# mask = np.zeros((640, 480))
# mask[310,510] = 1
mask[300:320, 500:520] = 1
mask[150:200, 350:400] = 1
# mask = np.array([510,310])


cv2.grabCut(img,mask,None,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
cv2.imshow("grabcut", img)
cv2.waitKey(0)
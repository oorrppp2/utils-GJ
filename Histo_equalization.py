import numpy as np
import cv2
from PIL import Image

depth = np.array(Image.open("/home/user/python_projects/DenseFusion_yolact_base/living_lab_video/17/depth_image28.png"))
depth = np.uint8(depth)
depth.astype(np.uint8)
print(depth)

# depth = cv2.cvtColor(depth, cv2.IMREAD_GRAYSCALE)

# depth = cv2.imread("/home/user/python_projects/DenseFusion_yolact_base/living_lab_video/17/depth_image28.png", cv2.IMREAD_GRAYSCALE)
# print(depth)
img_numpy = cv2.imread("/home/user/python_projects/DenseFusion_yolact_base/living_lab_video/17/color_image28.png")
img_numpy.astype(np.float32)
img_numpy0 = img_numpy[:,:,0].copy()
img_numpy[:,:,0] = img_numpy[:,:,2]
img_numpy[:,:,2] = img_numpy0
# cv2.im
equ_depth = depth.copy()
equ_depth = cv2.equalizeHist(depth, depth)

cv2.imwrite("/home/user/color.png", img_numpy)
cv2.imwrite("/home/user/depth.png", equ_depth)

# cv2.imshow("yolact detection", depth)
# cv2.waitKey(0)
import cv2

depth = cv2.imread("/home/user/NOCS_test_imgs/depth_image0.png", -1)
print(depth)
print(depth.shape)
import cv2
import numpy as np

img1 = cv2.imread("/home/user/Reference_images/005_tomato_soup_can_2.png", 1)
img2 = cv2.imread("/home/user/Reference_images/005_tomato_soup_can_3.png", 1)
img3 = cv2.imread("/home/user/Reference_images/005_tomato_soup_can_4.png", 1)
img4 = cv2.imread("/home/user/Reference_images/005_tomato_soup_can_5.png", 1)

cv2.imshow("img1", img1)

connected_img = cv2.hconcat(img1, img2)
cv2.imshow("connected_img", connected_img)
cv2.waitKey(0)
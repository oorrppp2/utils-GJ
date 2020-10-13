import numpy as np
import cv2

img = np.zeros((480, 640))
row, col = img.shape

original_img = cv2.imread("/home/user/sample_image/image171_color.png")

# cup = cv2.imread("/home/user/label/color_image171_cup.png")
spam = cv2.imread("/home/user/label/color_image0_spam.png")
tuna = cv2.imread("/home/user/label/color_image0_tuna.png")

# cup = cv2.cvtColor(cup, cv2.COLOR_BGR2GRAY)
spam = cv2.cvtColor(spam, cv2.COLOR_BGR2GRAY)
tuna = cv2.cvtColor(tuna, cv2.COLOR_BGR2GRAY)
# print(img)
max_val = 0
for i in range(row):
    for j in range(col):
        if tuna[i][j] == 255:
            img[i][j] = 6
        if spam[i][j] == 255:
            img[i][j] = 9
        # if cup[i][j] == 255:
        #     img[i][j] = 14

label = img
row,col = label.shape
labels = []
for i in range(row):
    for j in range(col):
        # if j > 500:
        #     img[i][j] = 0
        if label[i][j] != 0:
            # img[i][j] = color[label[i][j]-1]
            if label[i][j] not in labels:
                labels.append(label[i][j])

print(labels)

cv2.imwrite("/home/user/label.png", img)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("fish_can", tuna)
# cv2.imshow("mug", cup)
# cv2.imshow("spam", spam)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread("/home/user/real_scene_color2.png")
#
# rois = np.zeros((3,4))
#
# rois[0] = [134, 349,  16, 205]
# rois[1] = [116, 232, 209, 346]
# rois[2] = [102, 297, 413, 608]
#
# print(rois)
#
# rois = rois.astype(np.int)
#
# for i in range(3):
#     cv2.rectangle(img, (rois[i,2], rois[i,0]), (rois[i,3], rois[i,1]), (0, 0, 255), 2)
#
# cv2.imshow("img", img)
# cv2.waitKey(0)

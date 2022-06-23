import numpy as np
import cv2

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
        "013_apple_google_16k"      # 22
        "029_plate_google_16k"      # 23
        "030_fork_google_16k"       # 24
        "031_spoon_google_16k"      # 25
        "032_knife_google_16k"      # 26
]

img = np.zeros((480, 640))
row, col = img.shape

original_img = cv2.imread("/home/user/ycb_test_images/color_1010.png")

apple = cv2.imread("/home/user/ycb_test_images/color_1010_apple.png")
banana = cv2.imread("/home/user/ycb_test_images/color_1010_banana.png")
bowl = cv2.imread("/home/user/ycb_test_images/color_1010_bowl.png")
cup = cv2.imread("/home/user/ycb_test_images/color_1010_cup.png")
fork = cv2.imread("/home/user/ycb_test_images/color_1010_fork.png")
knife = cv2.imread("/home/user/ycb_test_images/color_1010_knife.png")
spoon = cv2.imread("/home/user/ycb_test_images/color_1010_spoon.png")

original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)
banana = cv2.cvtColor(banana, cv2.COLOR_BGR2GRAY)
bowl = cv2.cvtColor(bowl, cv2.COLOR_BGR2GRAY)
cup = cv2.cvtColor(cup, cv2.COLOR_BGR2GRAY)
fork = cv2.cvtColor(fork, cv2.COLOR_BGR2GRAY)
knife = cv2.cvtColor(knife, cv2.COLOR_BGR2GRAY)
spoon = cv2.cvtColor(spoon, cv2.COLOR_BGR2GRAY)

for i in range(row):
    for j in range(col):
        if apple[i][j] == 255:
            img[i][j] = 22
        if banana[i][j] == 255:
            img[i][j] = 10
        if bowl[i][j] == 255:
            img[i][j] = 13
        if cup[i][j] == 255:
            img[i][j] = 14
        if fork[i][j] == 255:
            img[i][j] = 24
        if knife[i][j] == 255:
            img[i][j] = 26
        if spoon[i][j] == 255:
            img[i][j] = 25

dellist = [original_img == 255]
# print(dellist)
img[dellist] = 0

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

cv2.imwrite("/home/user/ycb_test_images/label_1010.png", img)
# cv2.imshow("/home/user/label.png", img)
# cv2.waitKey(0)


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

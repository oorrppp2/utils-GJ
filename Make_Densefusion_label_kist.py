import numpy as np
import cv2

img_number = 0

img = np.zeros((480, 640))
row, col = img.shape

original_img = cv2.imread("/home/user/6D_pose_estimation_kist_dataset/color_" + str(img_number) + ".png")

# cup = cv2.imread("/home/user/label/color_image171_cup.png")
# root_dir = '/home/user/6D_pose_estimation_labeled_dataset/color_'
root_dir = '/media/user/ssd_1TB/Virtualbox/shared_folder/zipsa_6d_pose/color_'
apple = cv2.imread(root_dir + str(img_number) + "_apple.png")
bottle = cv2.imread(root_dir + str(img_number) + "_bottle.png")
bowl = cv2.imread(root_dir + str(img_number) + "_bowl.png")
fork = cv2.imread(root_dir + str(img_number) + "_fork.png")
knife = cv2.imread(root_dir + str(img_number) + "_knife.png")
spoon = cv2.imread(root_dir + str(img_number) + "_spoon.png")
plate = cv2.imread(root_dir + str(img_number) + "_plate.png")
banana1 = cv2.imread(root_dir + str(img_number) + "_banana1.png")
banana2 = cv2.imread(root_dir + str(img_number) + "_banana2.png")
mug = cv2.imread(root_dir + str(img_number) + "_mug.png")


# tomato = cv2.cvtColor(tomato, cv2.COLOR_BGR2GRAY)
# mustard = cv2.cvtColor(mustard, cv2.COLOR_BGR2GRAY)

apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)
bottle = cv2.cvtColor(bottle, cv2.COLOR_BGR2GRAY)
bowl = cv2.cvtColor(bowl, cv2.COLOR_BGR2GRAY)
fork = cv2.cvtColor(fork, cv2.COLOR_BGR2GRAY)
knife = cv2.cvtColor(knife, cv2.COLOR_BGR2GRAY)
spoon = cv2.cvtColor(spoon, cv2.COLOR_BGR2GRAY)
plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
banana1 = cv2.cvtColor(banana1, cv2.COLOR_BGR2GRAY)
banana2 = cv2.cvtColor(banana2, cv2.COLOR_BGR2GRAY)
mug = cv2.cvtColor(mug, cv2.COLOR_BGR2GRAY)
# print(img)
max_val = 0
for i in range(row):
    for j in range(col):
        if banana1[i][j] == 255:
            img[i][j] = 1
        if banana2[i][j] == 255:
            img[i][j] = 10
        if apple[i][j] == 255:
            img[i][j] = 2
        if bowl[i][j] == 255:
            img[i][j] = 3
        if mug[i][j] == 255:
            img[i][j] = 4
        if plate[i][j] == 255:
            img[i][j] = 5
        if fork[i][j] == 255:
            img[i][j] = 6
        if spoon[i][j] == 255:
            img[i][j] = 7
        if knife[i][j] == 255:
            img[i][j] = 8
        if bottle[i][j] == 255:
            img[i][j] = 9

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
#
cv2.imwrite("/home/user/6D_pose_estimation_labeled_dataset/label_" + str(img_number) + ".png", img)
# cv2.imshow("label", img)
# cv2.waitKey(0)
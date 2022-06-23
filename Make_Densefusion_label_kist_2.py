import numpy as np
import cv2

img_number = 0

img = np.zeros((480, 640))
row, col = img.shape

# original_img = cv2.imread("/home/user/ZiPSA_6D_pose_estimation/scene5/7709-")

# cup = cv2.imread("/home/user/label/color_image171_cup.png")
# root_dir = '/home/user/6D_pose_estimation_labeled_dataset/color_'
root_dir = "/home/user/ZiPSA_6D_pose_estimation/scene5/13516-"
# bottle = cv2.imread(root_dir  + "juice.png")
bowl = cv2.imread(root_dir + "bowl.png")
# tomato = cv2.imread(root_dir  + "tomato.png")
mug = cv2.imread(root_dir +"mug.png")
mustard = cv2.imread(root_dir  + "mustard.png")
coca = cv2.imread(root_dir +  "coca.png")


# tomato = cv2.cvtColor(tomato, cv2.COLOR_BGR2GRAY)
# mustard = cv2.cvtColor(mustard, cv2.COLOR_BGR2GRAY)

# bottle = cv2.cvtColor(bottle, cv2.COLOR_BGR2GRAY)
bowl = cv2.cvtColor(bowl, cv2.COLOR_BGR2GRAY)
# tomato = cv2.cvtColor(tomato, cv2.COLOR_BGR2GRAY)
mug = cv2.cvtColor(mug, cv2.COLOR_BGR2GRAY)
mustard = cv2.cvtColor(mustard, cv2.COLOR_BGR2GRAY)
coca = cv2.cvtColor(coca, cv2.COLOR_BGR2GRAY)
# print(img)
max_val = 0
for i in range(row):
    for j in range(col):
        if bowl[i][j] == 255:
            img[i][j] = 13
        if mug[i][j] == 255:
            img[i][j] = 14
        # if tomato[i][j] == 255:
        #     img[i][j] = 4
        # if bottle[i][j] == 255:
        #     img[i][j] = 22
        if coca[i][j] == 255:
            img[i][j] = 23
        if mustard[i][j] == 255:
            img[i][j] = 5

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
cv2.imwrite(root_dir + "label.png", img)
cv2.imshow("label", img)
cv2.waitKey(0)
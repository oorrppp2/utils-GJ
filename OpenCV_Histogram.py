import cv2

# img = cv2.imread('data/재학증명서.jpg', cv2.IMREAD_GRAYSCALE)
img_gray = cv2.imread('ychan/1.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('data/예금구좌신고서.jpg', -1)
img = img_gray
row, col = img.shape

# row = int(row*0.3)
# col = int(col*0.3)
# img = cv2.resize(img, (int(col), int(row)))


print("shape : " + str(img.shape))
# cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print("shape : " + str(img.shape))
# print(img)

for y in range(row):
    for x in range(col):
        if img[y,x] < 150:
            pass
            # img[y,x] = 0
        else:
            img[y,x] = 255

# row = int(row*0.3)
# col = int(col*0.3)
# img = cv2.resize(img, (int(col), int(row)))

# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("/home/user/result.jpg", img)
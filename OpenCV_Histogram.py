import cv2

img = cv2.imread('application.jpg', cv2.IMREAD_GRAYSCALE)
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
            img[y,x] = 0
        else:
            img[y,x] = 255

# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("histo.jpg", img)
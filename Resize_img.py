import cv2

filename = "라즈베리파이_출력창.jpg"
img = cv2.imread('/media/user/ssd_1TB/Virtualbox/shared_folder/'+filename)

row, col, _ = img.shape
row = int(row*0.5)
col = int(col*0.5)
img = cv2.resize(img, (int(col), int(row)))


print("shape : " + str(img.shape))

# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("/home/user/"+filename, img)
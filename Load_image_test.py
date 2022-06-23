import cv2

depth = cv2.imread("/home/user/catkin_ws/YCB_dataset_from_gazebo/image/depth_"+str(200)+".png", -1)
print(depth.shape)
print(depth)
cv2.imshow("depth", depth*1000)
cv2.waitKey(0)
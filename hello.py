# import numpy as np
import math
import cv2
# from scipy import io
# print("Hi")
#
# # a = np.zeros((2,3))
# # a = [[1,2,3],
# #      [4,5,6],
# #      [7,8,9],
# #      [2,3,4]]
# # a = np.array(a)
# # print("a")
# # print(a)
# #
# # print("shape : " + str(a.shape))
# #
# # # 행 바꾸기.
# # # a0 = a[:,0].copy()
# # # a[:,0] = a[:,1]
# # # a[:,1] = a0
# # print("a.T")
# # print(a.T)
# # print("shape : " + str(a.T.shape))
# #
# # # 모든 열을 2행 값으로 나누기
# # a = (a.T / a.T[2,:]).T
# # # a = a / a.T[:,2]
# #
# # print(a)
# #
# #
# # image = cv2.imread("/home/user/0001.png")
# # print(str(type(image)))
# # print(str(image.shape))
# # img = image[:,:,:3]
# # cv2.imshow("img", img)
# # cv2.waitKey(0)
#
# # K = [[240,320,3],
# #      [241,321,6],
# #      [239,319,9],
# #      [240,321,4],
# #      [240,319,4],
# #      [241,320,4],
# #      [241,319,4],
# #      [239,321,4],
# #      [239,320,4]]
# # K = np.array(K)
# # image[K[:,0],K[:,1],0] = 0
# # image[K[:,0],K[:,1],1] = 0
# # image[K[:,0],K[:,1],2] = 255
# #
# # cv2.imshow("img", image)
# # cv2.waitKey()
#
# # P = [1,2,3,4,5]
# # print(P[int(2.3)])
#
# # float_img = np.random.random((40,40))
# # # print(float_img)
# # uint_img = np.array(float_img*255).astype('uint8')
# # # print(uint_img)
# # grayImage = cv2.cvtColor(float_img, cv2.COLOR_GRAY2BGR)
# # # im = np.array(grayImage * 255, dtype = np.uint8)
# # # print(str(type(uint_img)))
# # # print(str(uint_img.shape))
# # # print(uint_img)
# # cv2.imshow("img", grayImage)
# # cv2.waitKey(0)
#
# # mat_file = io.loadmat('/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_dataset/data/0006/000001-meta.mat')
# # while 1:
# #     data = mat_file.popitem()
# #     if data is None:
# #         print("end of file")
# #         break
# #     else:
# #         if data[0] == 'vertmap':
# #             print("size : " + str(data[1].shape))
# #             cv2.imshow("img", data[1])
# #             cv2.waitKey(0)
# #             # print(data[1])
#
# # for i in range(10,20):
# #     print(i)
#
# # img = cv2.imread('/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_dataset/data/0001/000001-label.png')
# # img = img[:,:,0]
# # r = []
# # for i in range(480):
# #     for j in range(640):
# #         if img[i][j] != 0:
# #             if img[i][j] in r:
# #                 continue
# #             else:
# #                 r.append(img[i][j])
# #
# # print(r)
#
# # arr = []
# # arr.append(2)
# # arr.append(4)
# # arr.append(15)
# # print(arr)
# # print("arr back : " + str(arr[len(arr)-1]))
#
# img = np.zeros((480, 640))
# row, col = img.shape
#
# fish_can = cv2.imread("/home/user/label/fish_can.png")
# mug = cv2.imread("/home/user/label/mug.png")
# spam = cv2.imread("/home/user/label/spam.png")
#
# fish_can = cv2.cvtColor(fish_can, cv2.COLOR_BGR2GRAY)
# mug = cv2.cvtColor(mug, cv2.COLOR_BGR2GRAY)
# spam = cv2.cvtColor(spam, cv2.COLOR_BGR2GRAY)
# # print(img)
# max_val = 0
# for i in range(row):
#     for j in range(col):
#         if fish_can[i][j] == 255:
#             img[i][j] = 6
#         if mug[i][j] == 255:
#             img[i][j] = 14
#         if spam[i][j] == 255:
#             img[i][j] = 9
# cv2.imwrite("/home/user/label_float.png", img)
#
# label = img
# # label = cv2.imread("/home/user/label.png")
# # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
# row,col = label.shape
# labels = []
# for i in range(row):
#     for j in range(col):
#         if label[i][j] != 0:
#             # img[i][j] = color[label[i][j]-1]
#             if label[i][j] not in labels:
#                 labels.append(label[i][j])
#
# print(labels)
#
# # cv2.imshow("fish_can", fish_can)
# # cv2.imshow("mug", mug)
# # cv2.imshow("spam", spam)
# # cv2.imshow("img", img)
# # cv2.waitKey(0)
#
# # img = cv2.imread("/home/user/real_scene_color2.png")
# #
# # rois = np.zeros((3,4))
# #
# # rois[0] = [134, 349,  16, 205]
# # rois[1] = [116, 232, 209, 346]
# # rois[2] = [102, 297, 413, 608]
# #
# # print(rois)
# #
# # rois = rois.astype(np.int)
# #
# # for i in range(3):
# #     cv2.rectangle(img, (rois[i,2], rois[i,0]), (rois[i,3], rois[i,1]), (0, 0, 255), 2)
# #
# # cv2.imshow("img", img)
# # cv2.waitKey(0)

# import random
# index = int(random.random() * 1000)
# print(index)


"""
    torch test
"""
# import torch
# x_list = [0,1,2]
# x = torch.tensor(x_list)
# # x = torch.zeros((1))
# # # x += torch.mean()
# print(x)
# print(x.size)
# exit()

# x = torch.tensor([[1,2,3],
#                   [2,3,1],
#                   [2,2,2],
#                   [3,2,1]])
# print(x)
# print(x.size())
# rm_index = 3
# x = torch.cat((x[:rm_index,:], x[rm_index+1:,:]))
# # x = torch.cat((x[:,:rm_index], x[:,rm_index+1:]))
# print(x)
# print(x.size())
# x = x.repeat(1,5,1)
# print(x)
# print(x.size())



# import cv2
# img = cv2.imread("/media/user/ssd_1TB/disinfect/SUNRGBD-disinfect/images/kinect2data__002789_2014-06-22_19-36-20_094959634447_rgbf000086-resize.jpg")
# txt = open("/media/user/ssd_1TB/disinfect/SUNRGBD-disinfect/images/kinect2data__002789_2014-06-22_19-36-20_094959634447_rgbf000086-resize.txt", 'r')
# read_lines = txt.readlines()
# txt.close()
# for line in read_lines:
#     read_line = line
#     print(line)
#     print(img.shape)
#     x_center = float(read_line.split()[1])
#     y_center = float(read_line.split()[2])
#     width = float(read_line.split()[3])
#     height = float(read_line.split()[4])
#     int_x_center = int(img.shape[1]*x_center)
#     int_y_center = int(img.shape[0]*y_center)
#     int_width = int(img.shape[1]*width)
#     int_height = int(img.shape[0]*height)
#
#     cv2.rectangle(img, (int(int_x_center-int_width/2), int(int_y_center-int_height/2)), (int(int_x_center+int_width/2), int(int_y_center+int_height/2)), (0, 0, 255), 2)
#     cv2.imshow("img", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# obj = np.array([1,5,3,8,9])
# dim = obj.size
# int_obj = 0
# for i in range(dim):
#     int_obj += np.power(10, i) * obj[dim-1-i]
# print(int_obj)
#
# print(int(np.log10(int_obj)) + 1 if int_obj else 0)
#
# dim = int(np.log10(int_obj)) + 1 if int_obj else 0
# # arr_obj = np.array((0))
# arr_obj = []
# # print(arr_obj)
# for j in range(dim):
#     # print(int_obj%np.power(10,dim-j))
#     arr_obj.append(int(int_obj%np.power(10,dim-j)/np.power(10,dim-1-j)))
#     # np.append(arr_obj, 1, axis=0)
# print(arr_obj)
#
# for j in range(3):
#     print(j)

# def Range(end_num=0, start_num=0, interval=1):

def Range(*num):
    if len(num) == 1:
        end_num = num[0]
        start_num = 0
        interval = 1
    elif len(num) == 2:
        start_num = num[0]
        end_num = num[1]
        interval = 1
    elif len(num) == 3:
        start_num = num[0]
        end_num = num[1]
        interval = num[2]
    i = start_num
    return_list = []
    while i < end_num:
        return_list.append(i)
        i += interval
    return return_list

def get_loc1():
    return (20, 30)

def get_loc2():
    return {'x':20, 'y':30}

loc = get_loc1()
print('x:', loc[0], ' y:', loc[1])
loc = get_loc2()
print('x:', loc['x'], ' y:', loc['y'])

def hms(sec):
    s = ''
    if sec / 3600 > 0:
        s += str(int(sec / 3600)) + ':'
        sec %= 3600
    s += str(int(sec / 600))
    sec %= 600
    print(sec)
    s += str(int(sec / 60)) + ':'
    sec %= 60
    s += str(int(sec / 100))
    sec %= 10
    s += str(sec % 60)
    return s
print(hms(3601))

def my_max(*n):
    import numpy as np
    max_val = -np.inf
    for i in n:
        max_val = i if i > max_val else max_val
    return max_val
print(my_max(1, 9, 4, 3))

def max_count(l=[]):
    import numpy as np
    max_count = -np.inf
    for i in l:
        count = l.count(i)
        max_count = count if count > max_count else max_count
    return max_count
print(max_count([1,3,4,3,1,1, 3,33,3,3,3,3,3,3,3]))

f = ["apple", "banana", "cherry"]
for x in f:
    print(x, end=";")

class R:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def area(self):
        return self.width * self.height

class S(R):
    super(R)
    # def __init__(self):
    #     super(R, self)

# print(S)

n = [2,3,5]
print()
print([i**2 for i in n])

# print(Range(1, 8, 2))


# def max_first(x,y):
#     return [x,y] if x>y else [y,x]
# a = [3,4]
# b = max_first(a[0], a[1])
# # print(a,b)
#
# class Pro:
#     @property
#     def x(self):
#         print('get:', self.__x )
#         return self.__x
#     @x.setter
#     def x(self,x):
#         print('set:', x)
#         self.__x = x
#     def __init__(self, x):
#         if x > 0:
#             self.__x = x
#         else:
#             self.x = 0
#     def __str__(self):
#         return "x="+str(self.x)
# test = Pro(6)
# test.x = test.x+2
# test = Pro(0)
# test.x += 2
# print(test)
# import BTC
# x = 0

# input(x)
# print("x : " + str(x))

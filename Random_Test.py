import numpy as np
from random import *
cnt = 0
for i in range(260000):
    f = random()
    if f > 0.999:
       cnt += 1

# print(cnt)

array = np.empty((0))
print(type(array))
print(array.shape)
print(array)

append_array = np.zeros((5,1))
append_array[0,0] = 1
append_array[1,0] = 2
append_array[2,0] = 3
append_array[3,0] = 4
append_array[4,0] = 5

array = np.append(array, append_array)
print(array)

append_array = np.zeros((3,1))
append_array[0,0] = 9
append_array[1,0] = 8
append_array[2,0] = 7

array = np.append(array, append_array)
print(array)
import time
# import numpy as np


# m = 0
# t = time.time()
# a = [np.random.rand(1) for _ in range(1000000)]
# print(max(a))
# print(time.time() - t)
# t = time.time()
# for i in range(len(a)):
#     if a[i] > m:
#         m = a[i]

# print(m)
# print(time.time() - t)


# m = 0
# t = time.time()
# a = [np.random.rand(1) for _ in range(1000000)]
# print(max(a))
# print(time.time() - t)
# t = time.time()
# for i in range(len(a)):
#     if a[i] > m:
#         m = a[i]

# print(m)
# print(time.time() - t)


# from sys import stdin
# from queue import PriorityQueue

# s = PriorityQueue() # max is top
# l = PriorityQueue() # min is top

# for i in range(100):
#     s.put(np.random.rand(1))
# t = time.time()
# print(s.qsize())
# print(time.time() - t)

# for i in range(1000000):
#     l.put(np.random.rand(1))
# t = time.time()
# print(l.qsize())
# print(time.time() - t)


# size = 3000
# l_2D = [[i for i in range(size)] for j in range(size)]
# l_1D = [i for i in range(size*size)]

# t = time.time()
# for i in range(size):
#     for j in range(size):
#         l_2D[i][j] += 1

# print(time.time() - t)
# t = time.time()

# for i in range(size*size):
#     l_1D[i] += 1
# print(time.time() - t)

# s = set()
# l = [[1, 3, 5, 5, 7, 7, 9] for i in range(10)]
# for i in range(10):
#     ls = [set(l[i])]
#     print(ls)
#     s.add(ls)
# print(l)
# print(s)
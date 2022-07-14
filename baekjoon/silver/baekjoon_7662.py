# Baekjoon 7662

from sys import stdin
from collections import deque

# N, M = list(map(int, stdin.readline().split()))
# apple_box = []
queue = deque() # h, r, c, days
queue.append(2)
queue.append(3)
queue.append(4)
queue.append(5)

a = queue.prev
print(a)
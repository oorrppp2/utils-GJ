# Baekjoon 11279

from sys import stdin
from queue import PriorityQueue

N = int(stdin.readline())
heap = PriorityQueue()

for i in range(N):
    n = int(stdin.readline())
    if n == 0:
        if heap.empty():
            print(0)
        else:
            print(-heap.get())
    else:
        heap.put(-n)

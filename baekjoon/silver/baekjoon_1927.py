# Baekjoon 1927

from sys import stdin
import heapq

N = int(stdin.readline())

heap = []
for i in range(N):
    n = int(stdin.readline())
    if n != 0:
        heapq.heappush(heap, n)
    else:
        if len(heap) > 0:
            print(heapq.heappop(heap))
        else:
            print(0)

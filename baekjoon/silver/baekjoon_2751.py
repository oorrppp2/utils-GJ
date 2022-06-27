# Baekjoon 2751

from queue import PriorityQueue
from sys import stdin
N = int(stdin.readline())

p_queue = PriorityQueue()
for i in range(N):
    p_queue.put(int(stdin.readline()))

# print(p_queue)
while not p_queue.empty():
    print(p_queue.get())
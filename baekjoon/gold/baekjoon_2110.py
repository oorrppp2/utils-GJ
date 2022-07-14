# Baekjoon 2110

from sys import stdin
from queue import PriorityQueue

N, C = list(map(int, stdin.readline().split()))
position = []
for i in range(N):
    pos = int(stdin.readline())
    position.append(pos)
position.sort()
# print(position)

left = 0
right = position[N-1]

while left <= right:
    mid = (right + left) // 2
    pre_i = 0
    cnt = 1
    for i in range(1, N):
        if position[i] - position[pre_i] >= mid:
            pre_i = i 
            cnt += 1
    if cnt >= C:    # 기준 value를 높혀서 총 공유기의 수가 작아지게 해야됨 
        left = mid + 1
    else:
        right = mid - 1

print(right)
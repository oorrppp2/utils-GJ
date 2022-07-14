# Baekjoon 7576

from sys import stdin
from collections import deque

N, M = list(map(int, stdin.readline().split()))
apple_box = []
queue = deque() # h, r, c, days
visited = [[0 for _ in range(N)] for _ in range(M)]

for i in range(M):
    m = list(map(int, stdin.readline().split()))
    for k in range(len(m)):
        if m[k] == 1:
            queue.append([i, k, 0])
            visited[i][k] = 1
        elif m[k] == -1:
            visited[i][k] = 1
    apple_box.append(m)
    
last_day = 0
while len(queue) > 0:
    current = queue.popleft()
    r, c, days = current[0], current[1], current[2]
    if days > last_day:
        last_day = days
    if r > 0:
        if visited[r-1][c] != 1 and apple_box[r-1][c] == 0:
            queue.append([r-1, c, days+1])
            visited[r-1][c] = 1
    if c > 0:
        if visited[r][c-1] != 1 and apple_box[r][c-1] == 0:
            queue.append([r, c-1, days+1])
            visited[r][c-1] = 1

    if r < M-1:
        if visited[r+1][c] != 1 and apple_box[r+1][c] == 0:
            queue.append([r+1, c, days+1])
            visited[r+1][c] = 1
    if c < N-1:
        if visited[r][c+1] != 1 and apple_box[r][c+1] == 0:
            queue.append([r, c+1, days+1])
            visited[r][c+1] = 1
    
for r in range(M):
    for c in range(N):
        if visited[r][c] != 1:
            print(-1)
            exit(0)
print(last_day)
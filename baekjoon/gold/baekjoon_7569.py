# Baekjoon 7569

from sys import stdin
from collections import deque

N, M, H = list(map(int, stdin.readline().split()))
apple_box = []
queue = deque() # h, r, c, days
visited = [[[0 for _ in range(N)] for _ in range(M)] for _ in range(H)]

for h in range(H):
    apples = []
    for i in range(M):
        m = list(map(int, stdin.readline().split()))
        for k in range(len(m)):
            if m[k] == 1:
                queue.append([h, i, k, 0])
                visited[h][i][k] = 1
            elif m[k] == -1:
                visited[h][i][k] = 1
        apples.append(m)
    apple_box.append(apples)
    
last_day = 0
while len(queue) > 0:
    current = queue.popleft()
    h, r, c, days = current[0], current[1], current[2], current[3]
    if days > last_day:
        last_day = days
    if h > 0:
        if visited[h-1][r][c] != 1 and apple_box[h-1][r][c] == 0:
            queue.append([h-1, r, c, days+1])
            visited[h-1][r][c] = 1
    if r > 0:
        if visited[h][r-1][c] != 1 and apple_box[h][r-1][c] == 0:
            queue.append([h, r-1, c, days+1])
            visited[h][r-1][c] = 1
    if c > 0:
        if visited[h][r][c-1] != 1 and apple_box[h][r][c-1] == 0:
            queue.append([h, r, c-1, days+1])
            visited[h][r][c-1] = 1

    if h < H-1:
        if visited[h+1][r][c] != 1 and apple_box[h+1][r][c] == 0:
            queue.append([h+1, r, c, days+1])
            visited[h+1][r][c] = 1
    if r < M-1:
        if visited[h][r+1][c] != 1 and apple_box[h][r+1][c] == 0:
            queue.append([h, r+1, c, days+1])
            visited[h][r+1][c] = 1
    if c < N-1:
        if visited[h][r][c+1] != 1 and apple_box[h][r][c+1] == 0:
            queue.append([h, r, c+1, days+1])
            visited[h][r][c+1] = 1
    
    # for i in range(H):
    #     for j in range(M):
    #         print(visited[i][j])
    # print("===================================")
for h in range(H):
    for r in range(M):
        for c in range(N):
            if visited[h][r][c] != 1:
                print(-1)
                exit(0)
print(last_day)
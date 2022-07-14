# Baekjoon 1388

from sys import stdin
from queue import PriorityQueue


N, M = list(map(int, stdin.readline().split()))
visited = [[False for _ in range(M)] for _ in range(N)]
ans = 0
room = []
for i in range(N):
    room.append(stdin.readline().rstrip())

for i in range(N):
    for j in range(M):
        if not visited[i][j]:
            ans += 1
            visited[i][j] = True
            if room[i][j] == '-':
                for jj in range(j, M):
                    if room[i][jj] == '-':
                        visited[i][jj] = True
                    else:
                        break
            elif room[i][j] == '|':
                for ii in range(i, N):
                    if room[ii][j] == '|':
                        visited[ii][j] = True
                    else:
                        break
print(ans)

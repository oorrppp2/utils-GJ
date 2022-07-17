# Baekjoon 1325

from sys import stdin
from collections import deque

N, M = list(map(int, stdin.readline().split()))

trust = {}
ans = []

for i in range(N):
    trust[i+1] = []

for i in range(M):
    A, B = list(map(int, stdin.readline().split()))
    trust[B].append(A)

for B in range(N):
    count = 0
    computers = deque([B+1])
    visited = [False for _ in range(N)]
    visited[B] = True
    while computers:
        now = computers.popleft()
        for com in trust[now]:
            if not visited[com-1]:
                visited[com-1] = True
                computers.append(com)
                count += 1

    if count == 0:
        ans.append([-1, B+1])
    else:
        ans.append([-count, B+1])

ans.sort()
best = ans[0][0]
for i in range(len(ans)):
    if ans[i][0] == best:
        print(ans[i][1], end=' ')
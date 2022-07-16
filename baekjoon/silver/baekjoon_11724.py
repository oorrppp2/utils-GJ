# Baekjoon 11724

from sys import stdin
from queue import PriorityQueue

N, M = list(map(int, stdin.readline().split()))
neighbors = [[] for _ in range(N)]
visited = [False for _ in range(N)]
ans = 0

for i in range(M):
    p1, p2 = list(map(int, stdin.readline().split()))
    p1 -= 1
    p2 -= 1
    neighbors[p1].append(p2)
    neighbors[p2].append(p1)
for i in range(N):
    if len(neighbors[i]) > 0 and not visited[i]:
        queue = [i]
        visited[i] = True
        while len(queue) > 0:
            cur_node = queue.pop()
            if len(neighbors[cur_node]) > 0:
                for node in neighbors[cur_node]:
                    if not visited[node]:
                        queue.append(node)
                        visited[node] = True
        ans += 1
for v in visited:
    if not v:
        ans += 1
print(ans)
# Baekjoon 2606

from sys import stdin
N = int(stdin.readline())
M = int(stdin.readline())
graph = {}
visited = [False for _ in range(N)]
for i in range(N):
    graph[i+1] = []
for i in range(M):
    p1, p2 = list(map(int, stdin.readline().split()))
    graph[p1].append(p2)
    graph[p2].append(p1)

count = 1
queue = [1]
visited[0] = True
while len(queue) != 0:
    current = queue.pop(0)
    for node in graph[current]:
        if visited[node-1]:
            continue
        queue.append(node)
        visited[node-1] = True
        count += 1
print(count-1)
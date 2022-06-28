# Baekjoon 1667

from sys import stdin

N, K = list(map(int, stdin.readline().split()))
visited = [0 for _ in range(100001)]
if N == K:
    print(0)
    exit(0)
if N > K:
    print(N - K)
    exit(0)
queue = [[N, 0]]

while len(queue) != 0:
    pos, time = queue.pop(0)
    if pos - 1 == K or pos + 1 == K or pos * 2 == K:
        print(time+1)
        break
    if pos - 1 > 0:
        if visited[pos-1] == 0:
            queue.append([pos-1, time+1])
            visited[pos-1] = 1
    if pos + 1 < 100001:
        if visited[pos+1] == 0:
            queue.append([pos+1, time+1])
            visited[pos+1] = 1
    if pos * 2 < 100001:
        if visited[pos*2] == 0:
            queue.append([pos*2, time+1])
            visited[pos*2] = 1
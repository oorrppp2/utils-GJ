# Baekjoon 2644

from sys import stdin

n = int(stdin.readline())
x, y = list(map(int, stdin.readline().split()))
x -= 1
y -= 1
m = int(stdin.readline())

visited = [False for _ in range(n)]
chon = {}
for i in range(n):
    chon[i] = []

for i in range(m):
    h1, h2 = list(map(int, stdin.readline().split()))
    chon[h1-1].append(h2-1)
    chon[h2-1].append(h1-1)

queue = [[x, 0]]    # neighbor, visited num
visited[x] = True
ans = []
while len(queue) > 0:
    current = queue.pop(0)
    if current[0] == y:
        ans.append(current)
    neighbors = chon[current[0]]
    for neighbor in neighbors:
        if not visited[neighbor]:
            queue.append([neighbor, current[1]+1])
            visited[neighbor] = True

if len(ans) == 0:
    print(-1)
else:
    ans.sort()
    print(ans[0][1])
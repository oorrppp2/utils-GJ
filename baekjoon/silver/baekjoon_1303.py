# Baekjoon 1303

from sys import stdin

N, M = list(map(int, stdin.readline().split()))

ans = {}
ans['W'] = 0
ans['B'] = 0
soldiers = []
visited = [[False for _ in range(N)] for _ in range(M)]
for i in range(M):
    soldiers.append(stdin.readline().rstrip())

def search_dfs(color, row, col, nums):
    if row > 0:
        if not visited[row-1][col]:
            if soldiers[row-1][col] == color:
                visited[row-1][col] = True
                nums += search_dfs(color, row-1, col, 1)
    if row < M-1:
        if not visited[row+1][col]:
            if soldiers[row+1][col] == color:
                visited[row+1][col] = True
                nums += search_dfs(color, row+1, col, 1)
    if col > 0:
        if not visited[row][col-1]:
            if soldiers[row][col-1] == color:
                visited[row][col-1] = True
                nums += search_dfs(color, row, col-1, 1)
    if col < N-1:
        if not visited[row][col+1]:
            if soldiers[row][col+1] == color:
                visited[row][col+1] = True
                nums += search_dfs(color, row, col+1, 1)

    return nums

for i in range(M):
    for j in range(N):
        if not visited[i][j]:
            visited[i][j] = True
            nums = search_dfs(soldiers[i][j], i, j, 1)
            ans[soldiers[i][j]] += nums ** 2

print(ans['W'], ans['B'])
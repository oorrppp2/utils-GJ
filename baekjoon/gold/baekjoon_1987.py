# Baekjoon 1987

from sys import stdin
import time

R, C = list(map(int, stdin.readline().split()))

board = []
for i in range(R):
    board.append(stdin.readline().rstrip())

ans = 0
visited = [False for _ in range(26)]
visited[ord(board[0][0]) - ord('A')] = True
def dfs(row, col, count):
    global ans
    ans = max(ans, count)
    if row < R-1:
        if not visited[ord(board[row+1][col]) - ord('A')]:
            visited[ord(board[row+1][col]) - ord('A')] = True
            if count+1 == 26:
                print(26)
                exit(0)
            dfs(row+1, col, count+1)
            visited[ord(board[row+1][col]) - ord('A')] = False
    if col < C-1:
        if not visited[ord(board[row][col+1]) - ord('A')]:
            visited[ord(board[row][col+1]) - ord('A')] = True
            if count+1 == 26:
                print(26)
                exit(0)
            dfs(row, col+1, count+1)
            visited[ord(board[row][col+1]) - ord('A')] = False
    if row > 0:
        if not visited[ord(board[row-1][col]) - ord('A')]:
            visited[ord(board[row-1][col]) - ord('A')] = True
            if count+1 == 26:
                print(26)
                exit(0)
            dfs(row-1, col, count+1)
            visited[ord(board[row-1][col]) - ord('A')] = False
    if col > 0:
        if not visited[ord(board[row][col-1]) - ord('A')]:
            visited[ord(board[row][col-1]) - ord('A')] = True
            if count+1 == 26:
                print(26)
                exit(0)
            dfs(row, col-1, count+1)
            visited[ord(board[row][col-1]) - ord('A')] = False

dfs(0, 0, 1)
print(ans)
# Baekjoon 2178

from sys import stdin

N, M = list(map(int, stdin.readline().split()))
maze = []
occupied = [[0 for _ in range(M)] for _ in range(N)]
for i in range(N):
    maze.append(stdin.readline().split()[0])

queue = [[0,0,1]] # row, col, step
occupied[0][0] = 1
while len(queue) != 0:
    current = queue.pop(0)
    row = current[0]
    col = current[1]
    step = current[2]
    if row == N-1 and col == M-1:
        print(step)
        break
    if row > 0:
        if maze[row-1][col] == '1' and occupied[row-1][col] != 1:
            queue.append([row-1, col, step + 1])
            occupied[row-1][col] = 1
    if col > 0:
        if maze[row][col-1] == '1' and occupied[row][col-1] != 1:
            queue.append([row, col-1, step + 1])
            occupied[row][col-1] = 1
    if row < N-1:
        if maze[row+1][col] == '1' and occupied[row+1][col] != 1:
            queue.append([row+1, col, step + 1])
            occupied[row+1][col] = 1
    if col < M-1:
        if maze[row][col+1] == '1' and occupied[row][col+1] != 1:
            occupied[row][col+1] = 1
            queue.append([row, col+1, step + 1])
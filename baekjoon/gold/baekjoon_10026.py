# Baekjoon 10026

from sys import stdin

N = int(stdin.readline())
img = []
normal_grid = [[0 for _ in range(N)] for _ in range(N)]
non_normal_grid = [[0 for _ in range(N)] for _ in range(N)]
division_num = 1
for i in range(N):
    img.append(list(map(int, stdin.readline().split())))

working = [0, 0, img[0][0]]
normal_grid[0][0] = division_num
non_normal_grid[0][0] = division_num
# DFS
while True:
    if len(working) == 0:
        division_num += 1
        stop = False
        for i in range(N):
            if stop:
                break
            for j in range(N):
                if normal_grid[i][j] == 0:
                    working.append([i,j,img[i][j]])
                    break
    else:
        current = working.pop()
        r, c, color = current[0], current[1], current[2]
        if r+1 < N:
            if 
                if non_normal_grid[r+1][c] == 0:
                    working.append([r+1, c, img[r+1][c]])
                    non_normal_grid[r+1][c] = division_num
            elif color == 'R' or color == 'G':
                if non_normal_grid[r+1][c] == 0:
                    working.append([r+1, c, img[r+1][c]])
                    non_normal_grid[r+1][c] = division_num

# for i in range(N):
#     for j in range(N):
#         normal_grid[i][j] = division_num
#         non_normal_grid[i][j] = division_num
#         if i > 0:
#             if img[i][j] == img[i-1][j]:
#                 normal_grid[i][j] = normal_grid[i-1][j]
#                 non_normal_grid[i][j] = non_normal_grid[i-1][j]
#             elif (img[i][j] == 'R' and img[i-1][j] == 'G') or \
#                  (img[i][j] == 'G' and img[i-1][j] == 'R'):
#                 non_normal_grid[i][j] = non_normal_grid[i-1][j]
#         if i > 0:
#             if img[i][j] == img[i-1][j]:
#                 normal_grid[i][j] = normal_grid[i-1][j]
#                 non_normal_grid[i][j] = non_normal_grid[i-1][j]
#             elif (img[i][j] == 'R' and img[i-1][j] == 'G') or \
#                  (img[i][j] == 'G' and img[i-1][j] == 'R'):
#                 non_normal_grid[i][j] = non_normal_grid[i-1][j]


# Baekjoon 1018

from sys import stdin

M, N = list(map(int, stdin.readline().split()))
ans = []
board = []
for i in range(M):
    l = stdin.readline().split()[0]
    board.append(l)

for i in range(M-7):
    for j in range(N-7):
        B_left_top = 0
        W_left_top = 0
        for ii in range(i, i+8):
            l = board[ii]
            for jj in range(j, j+8):
                if ii % 2 == 0 and jj % 2 == 0:
                    if l[jj] != 'B':
                        B_left_top += 1
                    else:
                        W_left_top += 1
                elif ii % 2 == 0 and jj % 2 != 0:
                    if l[jj] != 'W':
                        B_left_top += 1
                    else:
                        W_left_top += 1
                elif ii % 2 != 0 and jj % 2 != 0:
                    if l[jj] != 'B':
                        B_left_top += 1
                    else:
                        W_left_top += 1
                elif ii % 2 != 0 and jj % 2 == 0:
                    if l[jj] != 'W':
                        B_left_top += 1
                    else:
                        W_left_top += 1
        ans.append(min([B_left_top, W_left_top]))

print(min(ans))



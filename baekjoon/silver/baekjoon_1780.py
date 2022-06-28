# Baekjoon 1780

from sys import stdin

M = int(stdin.readline())

paper = []
ans = {}
ans['-1'] = 0
ans['0'] = 0
ans['1'] = 0
for i in range(M):
    paper.append(list(map(int, stdin.readline().split())))

def solution(r_start, c_start, N):
    global ans
    global paper
    if N == 1:
        ans[str(paper[r_start][c_start])] += 1
        return
    none_of_them = True
    std = paper[r_start][c_start]
    stop_i = False
    for i in range(r_start, r_start+N):
        if stop_i:
            break
        for j in range(c_start, c_start+N):
            if paper[i][j] == std:
                pass
            else:
                stop_i = True
                break
    if not stop_i:
        ans[str(std)] += 1
        none_of_them = False
        return

    if none_of_them:
        solution(r_start, c_start, N//3)
        solution(r_start + N // 3, c_start, N//3)
        solution(r_start + N // 3*2, c_start, N//3)

        solution(r_start, c_start + N // 3, N//3)
        solution(r_start + N // 3, c_start + N // 3, N//3)
        solution(r_start + N // 3*2, c_start + N // 3, N//3)

        solution(r_start, c_start + N // 3 * 2, N//3)
        solution(r_start + N // 3, c_start + N // 3 * 2, N//3)
        solution(r_start + N // 3*2, c_start + N // 3 * 2, N//3)

solution(0, 0, M)
print(ans['-1'])
print(ans['0'])
print(ans['1'])
# Baekjoon 2630

from sys import stdin
N = int(stdin.readline())

paper = []
for i in range(N):
    paper.append(list(map(int, stdin.readline().split())))
# print(paper)

ans = [0, 0]
def slice_paper(r_start, c_start, size):
    global ans
    global paper
    if size == 1:
        ans[paper[r_start][c_start]] += 1
        return 
    else:
        std = paper[r_start][c_start]
        stop = False
        for r in range(r_start, r_start + size):
            for c in range(c_start, c_start + size):
                if paper[r][c] != std:
                    stop = True
                    break
            if stop:
                break
        if stop:
            slice_paper(r_start, c_start, size // 2)
            slice_paper(r_start, c_start+size//2, size // 2)
            slice_paper(r_start+size//2, c_start, size // 2)
            slice_paper(r_start+size//2, c_start+size//2, size // 2)
        else:
            ans[std] += 1

slice_paper(0, 0, N)
print(ans[0])
print(ans[1])
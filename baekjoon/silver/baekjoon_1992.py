# Baekjoon 1992

from sys import stdin

N = int(stdin.readline())

img = []
for i in range(N):
    s = stdin.readline().split()[0]
    img.append(s)
ans = []
def quadtree(element, r_start, c_start, S):
    global img
    std = img[r_start][c_start]
    if S == 1:
        return std
    stop = False
    for r in range(r_start, r_start + S):
        for c in range(c_start, c_start + S):
            if img[r][c] == std:
                pass
            else:
                stop = True
                break
        if stop:
            break
    if not stop:
        return std
    else:
        element += '('
        element += quadtree('', r_start, c_start, S//2)
        element += quadtree('', r_start, c_start+S//2, S//2)
        element += quadtree('', r_start+S//2, c_start, S//2)
        element += quadtree('', r_start+S//2, c_start+S//2, S//2)
        element += ')'
        return element

print(quadtree('', 0, 0, N))
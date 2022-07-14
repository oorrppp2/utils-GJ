# Baekjoon 2675

from sys import stdin
T = int(stdin.readline())
for i in range(T):
    R, S = list(stdin.readline().split())
    R = int(R)
    for s in S:
        for _ in range(R):
            print(s, end='')
    print()
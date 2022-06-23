# Baekjoon 1920

from sys import stdin
N = int(stdin.readline())
ans = []
A = set(map(int, stdin.readline().split()))

M = int(stdin.readline())
B = list(map(int, stdin.readline().split()))
for i in range(M):
    if B[i] in A:
        print(1)
        # ans.append(1)
    else:
        print(0)
        # ans.append(0)

# for a in ans:
#     print(a)
# Baekjoon 1929

from sys import stdin
M, N = list(map(int, stdin.readline().split()))
ans = [True] * (N+1)
ans[0] = False
ans[1] = False

for i in range(2, int(N**0.5)+1):
    if ans[i] == True:
        for j in range(i, N+1, i):
            ans[j] = False
        ans[i] = True

for i in range(M, len(ans)):
    if ans[i]:
        print(i)
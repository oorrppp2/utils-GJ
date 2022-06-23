# Baekjoon 15829

from sys import stdin
N = int(stdin.readline())
S = stdin.readline().split()[0]

ans = 0
M = 1234567891
for i in range(N):
    a_i = ord(S[i])-96
    for j in range(i):
        a_i *= 31
        a_i %= M
    ans += a_i
    ans %= M

print(ans)
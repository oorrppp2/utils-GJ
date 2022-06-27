# Baekjoon 1074

from sys import stdin

N, r, c = list(map(int, stdin.readline().split()))
ans = 0
for i in range(N, 0, -1):
    if r >= pow(2, i-1):
        r -= pow(2, i-1)
        ans += pow(2, 2*i-1)
    if c >= pow(2,i-1):
        c -= pow(2, i-1)
        ans += pow(2, 2*i-2)
print(ans)


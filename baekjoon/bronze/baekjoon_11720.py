# Baekjoon 11720

from sys import stdin
N = int(stdin.readline())
l = stdin.readline()
ans = 0
for i in range(N):
    ans += int(l[i])
print(ans)
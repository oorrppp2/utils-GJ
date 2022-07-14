# Baekjoon 5525

from sys import stdin

N = int(stdin.readline())
M = int(stdin.readline())

ans = 0
PN = 'I'
for i in range(N):
    PN += 'OI'

s = stdin.readline().split()[0]


i = 1
p = 0
while i < M-1:
    if s[i-1:i+2] == "IOI":
        p += 1
        if p == N:
            ans += 1
            p -= 1
        i += 2
    else:
        p = 0
        i += 1


print(ans)

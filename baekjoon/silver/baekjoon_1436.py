# Baekjoon 1436

from sys import stdin
N = int(stdin.readline())

ans = 0
i = 665
while True:
    if N == 0:
        ans = i
        break
    i += 1
    if '666' in str(i):
        N -= 1
        continue

print(ans)
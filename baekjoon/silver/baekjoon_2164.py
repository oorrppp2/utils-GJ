# Baekjoon 2164

from sys import stdin
N = int(stdin.readline())

for i in range(20):
    if pow(2, i) > N:
        N %= pow(2, i-1)
        break
    elif pow(2,i) == N:
        print(N)
        exit(0)
ans = N*2
print(ans)
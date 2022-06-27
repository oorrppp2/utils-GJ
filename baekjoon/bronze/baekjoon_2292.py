# Baekjoon 2292

from sys import stdin
N = int(stdin.readline())
n = 1
for i in range(10000000000):
    n += i*6
    if n >= N:
        print(i+1)
        break
# Baekjoon 2739

from sys import stdin
N = int(stdin.readline())
for i in range(9):
    print("{0} * {1} = {2}".format(N, i+1, N * (i+1)))

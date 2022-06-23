# Baekjoon 10952

from sys import stdin
# T = int(stdin.readline())
# for i in range(T):
while True:
    A, B = list(map(int, stdin.readline().split()))
    if A==0 and B==0:
        break
    print(A+B)

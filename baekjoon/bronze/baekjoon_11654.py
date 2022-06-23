# Baekjoon 11654

from sys import stdin
C = stdin.readline().split()[0]
try:
    N = int(C)
    print(ord(N))
except:
    print(ord(C))
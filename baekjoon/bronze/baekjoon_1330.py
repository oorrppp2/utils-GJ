# Baekjoon 1330

from sys import stdin
A, B = list(map(int, stdin.readline().split()))
if A == B:
    print('==')
else:
    print('<' if A < B else '>')
# Baekjoon 1269

from sys import stdin

AB_dict = {}
N, M = list(map(int, stdin.readline().split()))
A = set(map(int, stdin.readline().split()))
B = set(map(int, stdin.readline().split()))

print(len(A-B)+len(B-A))
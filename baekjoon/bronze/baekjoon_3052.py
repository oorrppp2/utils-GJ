# Baekjoon 3052

from sys import stdin
residual = []
count = 1
for i in range(10):
    n = int(stdin.readline())
    residual.append(n%42)
residual.sort()
for i in range(9):
    if residual[i+1] != residual[i]:
        count+=1

print(count)

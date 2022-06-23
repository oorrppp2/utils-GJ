# Baekjoon 2577

from sys import stdin
mul = 1
ans = [0]*10
for _ in range(3):
    N = int(stdin.readline())
    mul *= N
mul_s = str(mul)
for s in mul_s:
    ans[int(s)]+=1
for i in ans:
    print(i)
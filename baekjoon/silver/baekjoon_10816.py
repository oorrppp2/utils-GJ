# Baekjoon 10816

from sys import stdin
N = int(stdin.readline())
nums = list(map(int, stdin.readline().split()))
M = int(stdin.readline())
targets = list(map(int, stdin.readline().split()))

num_dict = {}
for n in nums:
    try:
        num_dict[n] += 1
    except:
        num_dict[n] = 1

for t in targets:
    try:
        print(num_dict[t], end=' ')
    except:
        print(0, end=' ')
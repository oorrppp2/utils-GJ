# Baekjoon 4673

from sys import stdin

self_nums = [0]*10001

for i in range(10000):
    str_i = str(i)
    self_num = i
    for n in str_i:
        self_num += int(n)
    if self_num <= 10000:
        self_nums[self_num] = 1

for i in range(10000):
    if self_nums[i] == 0:
        print(i)
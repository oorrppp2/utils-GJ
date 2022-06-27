# Baekjoon 10773

from sys import stdin
N = int(stdin.readline())

correct_nums = []
for i in range(N):
    in_num = int(stdin.readline())
    if in_num == 0:
        correct_nums.pop()
    else:
        correct_nums.append(in_num)
print(sum(correct_nums))
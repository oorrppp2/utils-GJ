# Baekjoon 2562

from sys import stdin
# N = int(stdin.readline())
max_value = 0
max_index = 0
for i in range(9):
    num = int(stdin.readline())
    if num > max_value:
        max_value = num
        max_index = i+1
print(max_value)
print(max_index)
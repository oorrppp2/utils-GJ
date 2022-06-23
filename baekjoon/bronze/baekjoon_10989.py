# Baekjoon 10989

from sys import stdin
N = int(stdin.readline())
count_sort_array = [0] * 10000
for i in range(N):
    num = int(stdin.readline())
    count_sort_array[num-1] += 1
for i in range(len(count_sort_array)):
    n = count_sort_array[i]
    for j in range(n):
        print(i+1)
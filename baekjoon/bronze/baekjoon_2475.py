# Baekjoon 2475

from sys import stdin
nums = list(map(int, stdin.readline().split()))
nums_sum = 0
for num in nums:
    nums_sum += num**2

print(nums_sum%10)
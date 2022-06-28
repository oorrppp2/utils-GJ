# Baekjoon 1065

from sys import stdin

N = int(stdin.readline())

han_su = 0
for i in range(1, N+1):
    nums = []
    str_n = str(i)
    for n in str_n:
        nums.append(int(n))
    
    if i < 10:
        han_su += 1
    else:
        diff = nums[1] - nums[0]
        is_hansu = True
        for j in range(1, len(nums)):
            if nums[j] - nums[j-1] != diff:
                is_hansu = False
                break
        if is_hansu:
            han_su += 1

print(han_su)
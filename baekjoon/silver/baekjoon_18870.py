# Baekjoon 18870

from sys import stdin
N = int(stdin.readline())
X = list(map(int, stdin.readline().split()))
nums = []
ans = [0] * N
grade = 0

for i in range(len(X)):
    nums.append([X[i], i])
nums.sort(key=lambda x:x[0])

for i in range(1, len(nums)):
    if nums[i][0] > nums[i-1][0]:
        grade += 1
    ans[nums[i][1]] = grade

for num in ans:
    print(num, end=' ')


# Baekjoon 18870 solution function
def solution(N, X):
    from sys import stdin
    nums = []
    ans = [0] * N
    grade = 0

    for i in range(len(X)):
        nums.append([X[i], i])
    nums.sort(key=lambda x:x[0])

    for i in range(1, len(nums)):
        if nums[i][0] > nums[i-1][0]:
            grade += 1
        ans[nums[i][1]] = grade
    
    return ans
    
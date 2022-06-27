# Baekjoon 1978

from sys import stdin
N = int(stdin.readline())
numbers = list(map(int, stdin.readline().split()))
ans = [True] * (1001)
ans[0] = False
ans[1] = False

for i in range(2, int(1001**0.5)+1):
    if ans[i] == True:
        for j in range(i, 1001, i):
            ans[j] = False
        ans[i] = True
prime_nums = 0
for n in numbers:
    if ans[n]:
        prime_nums += 1
print(prime_nums)
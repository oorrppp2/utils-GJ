# Baekjoon 11866

from sys import stdin
N, K = list(map(int, stdin.readline().split()))
circle = [i for i in range(1, N+1)]
elimination = 0
ans = []
for i in range(N):
    elimination += K-1
    elimination %= len(circle)
    ans.append(circle[elimination])
    circle = circle[:elimination] + circle[elimination+1:]

print("<", end='')
for i in range(len(ans)):
    if i == len(ans)-1:
        print(str(ans[i])+">")
    else:
        print(ans[i], end=', ')


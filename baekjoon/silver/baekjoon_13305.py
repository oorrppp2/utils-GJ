# Baekjoon 11399

# from sys import stdin
# N = int(stdin.readline())
# dist = list(map(int, stdin.readline().split()))
# cost = list(map(int, stdin.readline().split()))

# base = cost[0]
# base_i = 0
# i = 0
# ans = 0
# while i < N-2:
#     i += 1
#     if cost[i] < base:
#         ans += sum(dist[base_i:i]) * base
#         base_i = i
#         base = cost[i]

# ans += sum(dist[base_i:N]) * base
# print(ans)      # timeout

from sys import stdin
N = int(stdin.readline())
dist = list(map(int, stdin.readline().split()))
cost = list(map(int, stdin.readline().split()))

base = cost[0]
ans = 0
for i in range(N-1):
    if cost[i] < base:
        base = cost[i]
    ans += dist[i] * base

print(ans)
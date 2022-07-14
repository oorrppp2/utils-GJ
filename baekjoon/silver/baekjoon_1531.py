# # Baekjoon 1531

# from sys import stdin
# from queue import PriorityQueue


# N, M = list(map(int, stdin.readline().split()))
# ans = 0

# img = [[0 for _ in range(100)] for _ in range(100)]
# for t in range(N):
#     lx, ly, rx, ry = list(map(int, stdin.readline().split()))
#     for i in range(lx, rx+1):
#         for j in range(ly, ry+1):
#             img[i-1][j-1] += 1

# for i in range(100):
#     for j in range(100):
#         if img[i][j] <= M:
#             ans += 1

# print(10000-ans)


a = "123456"
# a[4] = "*"
a = a[:4] + "*" + a[5:]
print(a)
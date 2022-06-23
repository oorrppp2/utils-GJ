# Baekjoon 1931

from sys import stdin
N = int(stdin.readline())
end_time = 0
ans = 0
times = []
for i in range(N):
    start, end = list(map(int, stdin.readline().split()))
    times.append([start, end])
times.sort(key=lambda x:x[0])
times.sort(key=lambda x:x[1])
for i in range(len(times)):
    if times[i][0] >= end_time:
        end_time = times[i][1]
        ans += 1
print(ans)

# # Baekjoon 1931 solution function
# def solution(N, times):
#     from sys import stdin
#     end_time = 0
#     ans = 0
#     times.sort(key=lambda x:x[0])
#     times.sort(key=lambda x:x[1])
#     for i in range(len(times)):
#         if times[i][0] >= end_time:
#             end_time = times[i][1]
#             ans += 1
#     return ans
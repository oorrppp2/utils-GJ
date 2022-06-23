import sys

W, N = map(int, input().split())
S = [0 for i in range(1000)]
for i in range(N):
    M, P = map(int, input().split())
    S[P-1] += M
ans = 0
for i in range(1000, 0, -1):
    if S[i-1] == 0:
        continue
    elif S[i-1] <= W:
        W -= S[i-1]
        ans += S[i-1] * i
    else:
        ans += W * i
        break
print(ans)

# import random
# import time
# start = time.time()
# W = 1000
# N = 1000000
# S = [0 for i in range(1000)]
#
# for i in range(N):
#     M = random.randint(1, 1000)
#     P = random.randint(1, 1000)
#     S[P-1] += M
# ans = 0
# for i in range(1000, 0, -1):
#     if S[i-1] == 0:
#         continue
#     elif S[i-1] <= W:
#         W -= S[i-1]
#         ans += S[i-1] * i
#     else:
#         ans += W * i
#         print(ans)
#         break
# print("time :", time.time() - start)
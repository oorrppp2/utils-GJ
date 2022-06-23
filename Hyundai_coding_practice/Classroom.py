import sys
import heapq
import time
import random

start = time.time()

# N = int(input())
# H = []
# Q = []
# for i in range(N):
#     S, F = map(int, sys.stdin.readline().split())
#     heapq.heappush(H, [S, F])
# print(H)
# S, F = heapq.heappop(H)
# Q.append([S, F])
# for i in range(N-1):
#     S, F = heapq.heappop(H)
#     if F < Q[-1][1]:
#         Q.pop()
#         Q.append([S, F])
#     elif S >= Q[-1][1]:
#         Q.append([S, F])
# print(len(Q))

N = 1000000
H = []
Q = []
for i in range(N):
    # S, F = map(int, input().split())
    S = random.randint(1, 1000000000)
    F = random.randint(1, 1000000000)
    heapq.heappush(H, [S, F])
    # H.append([S, F])

print("time :", time.time() - start)
# exit()

S, F = heapq.heappop(H)
Q.append([S, F])
for i in range(N-1):
    S, F = heapq.heappop(H)
    # S, F = H.pop()
    if F < Q[-1][1]:
        Q.pop()
        Q.append([S, F])
    elif S >= Q[-1][1]:
        Q.append([S, F])
print(len(Q))

print("time :", time.time() - start)
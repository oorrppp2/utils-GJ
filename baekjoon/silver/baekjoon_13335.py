# Baekjoon 13335

from sys import stdin
from collections import deque

n, w, L = list(map(int, stdin.readline().split()))
trucks = list(map(int, stdin.readline().split()))

bridge = [[trucks.pop(0), w]]
ans = 1

while len(trucks) != 0:
    ans += 1
    residual_weight = L
    for i in range(len(bridge)):
        bridge[i][1] -= 1
        residual_weight -= bridge[i][0]
    if bridge[0][1] == 0:
        residual_weight += bridge[0][0]
        bridge.pop(0)

    if trucks[0] <= residual_weight:
        truck = trucks.pop(0)
        bridge.append([truck, w])

# Calculate residual trucks
ans += bridge[-1][1]

print(ans)
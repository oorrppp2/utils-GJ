# Baekjoon 18111

from sys import stdin
# N, M, B = list(map(int, stdin.readline().split()))
N, M, B = 500, 500, 0
ground = []
for i in range(N):
    # ground.extend(list(map(int, stdin.readline().split())))
    ground.extend([0 for i in range(500)])

ground[0] = 0
ground[1] = 25

ans = []
highest = max(ground)
lowest = min(ground)
best_cost = 1e+32
best_height = 0
for target in range(lowest, highest+1):
    cost = 0
    B_current = B

    for g in ground:
        diff = g-target
        # Digging fist
        if diff > 0:
            cost += 2*(diff)
        # Filling
        else:
            cost -= diff
        B_current += diff
    if B_current < 0:
        continue
    else:
        if cost <= best_cost:
            best_cost = cost
            best_height = target
print(best_cost, best_height)

# Baekjoon 1966

from sys import stdin
T = int(stdin.readline())

for i in range(T):
    N, M = list(map(int, stdin.readline().split()))
    queue = list(map(int, stdin.readline().split()))
    target = [False] * N
    target[M] = True
    order = 1
    while len(queue) != 0:
        if queue[0] < max(queue):
            queue.append(queue.pop(0))
            target.append(target.pop(0))
        else:
            if target[0]:
                print(order)
                break
            queue.pop(0)
            target.pop(0)
            order += 1

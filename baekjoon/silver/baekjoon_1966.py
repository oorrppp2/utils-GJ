# Baekjoon 1966

from sys import stdin
T = int(stdin.readline())

for i in range(T):
    N, M = list(map(int, stdin.readline().split()))
    queue = list(map(int, stdin.readline().split()))

    for j in range(len(queue)):
        if queue[j] < max(queue):
            queue.append(queue[j])
        else:
            if M == j:
                print(j+1)
                break
            # queue.pop(0)
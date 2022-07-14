# Baekjoon 5052

from sys import stdin
import heapq

T = int(stdin.readline())

for t in range(T):
    N = int(stdin.readline())
    numbers = []
    for n in range(N):
        numbers.append(stdin.readline().rstrip())

    numbers.sort(reverse=True)

    consistency = True
    while len(numbers) > 1:
        n1 = numbers.pop()
        n2 = numbers.pop()
        n1_size = len(n1)
        if n1 == n2[:n1_size]:
            consistency = False
            print('NO')
            break
        else:
            numbers.append(n2)

    if consistency:
        print('YES')
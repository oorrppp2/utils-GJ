# Baekjoon 9095

from sys import stdin
T = int(stdin.readline())

for t in range(T):
    N = int(stdin.readline())

    if N == 1:
        print(1)
    elif N == 2:
        print(2)
    elif N == 3:
        print(4)
    elif N == 4:
        print(7)

    else:
        combi = [1, 2, 3]
        ans = 0
        while len(combi) > 0:
            n = combi.pop()
            if n + 1 < N:
                combi.append(n+1)
            elif n+1 == N:
                ans += 1
            if n + 2 < N:
                combi.append(n+2)
            elif n+2 == N:
                ans += 1
            if n + 3 < N:
                combi.append(n+3)
            elif n+3 == N:
                ans += 1

        print(ans)
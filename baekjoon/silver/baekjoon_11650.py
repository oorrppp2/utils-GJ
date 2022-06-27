# Baekjoon 11650

from sys import stdin
N = int(stdin.readline())
coord = []
for i in range(N):
    xy = list(map(int, stdin.readline().split()))
    coord.append(xy)

coord = sorted(coord, key=lambda x:x[1])
coord = sorted(coord, key=lambda x:x[0])
for xy in coord:
    print(xy[0], xy[1])


# Baekjoon 11651

from sys import stdin
N = int(stdin.readline())
coord = []
for i in range(N):
    xy = list(map(int, stdin.readline().split()))
    coord.append(xy)

coord = sorted(coord, key=lambda x:x[0])
coord = sorted(coord, key=lambda x:x[1])
for xy in coord:
    print(xy[0], xy[1])

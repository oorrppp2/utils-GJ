# Baekjoon 10250

from sys import stdin
T = int(stdin.readline())
for i in range(T):
    H, W, sonnom = list(map(int, stdin.readline().split()))
    if sonnom % H == 0:
        print(str(H)+"{0:02d}".format((sonnom//H)))
    else:
        print(str(sonnom % H)+"{0:02d}".format((sonnom//H)+1))
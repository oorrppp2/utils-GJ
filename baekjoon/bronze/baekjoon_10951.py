# Baekjoon 10951

from sys import stdin
# T = int(stdin.readline())
# for i in range(T):
while True:
    try:
        A, B = list(map(int, stdin.readline().split()))
        print(A+B)
    except:
        break

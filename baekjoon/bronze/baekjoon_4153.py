# Baekjoon 4153

from sys import stdin

while True:
    s = list(map(int, stdin.readline().split()))
    if s == [0, 0, 0]:
        break

    s = sorted(s)
    if s[0] ** 2 + s[1] ** 2 == s[2] ** 2:
        print("right")
    else:
        print("wrong")

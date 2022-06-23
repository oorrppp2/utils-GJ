# Baekjoon 2884

from sys import stdin
H, M = list(map(int, stdin.readline().split()))
time_min = H*60+M + (24*60)
time_min -= 45
time_min %= (24*60)

H = int(time_min / 60)
M = time_min % 60
print(H, M)

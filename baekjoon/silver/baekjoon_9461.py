# Baekjoon 9461

from sys import stdin
T = int(stdin.readline())

padoban = [0 for _ in range(100)]
padoban[:6] = [1, 1, 1, 2, 2, 3]
for i in range(6, 100):
    padoban[i] = padoban[i-1] + padoban[i-5]

for t in range(T):
    N = int(stdin.readline())
    print(padoban[N-1])
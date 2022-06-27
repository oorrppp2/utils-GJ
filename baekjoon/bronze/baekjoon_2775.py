# Baekjoon 2775

from sys import stdin
T = int(stdin.readline())

apartment = [[i+1 for i in range(14)]]
for i in range(1, 15):
    i_th = []
    for j in range(14):
        i_th.append(sum(apartment[i-1][:j+1]))
    apartment.append(i_th)

for t in range(T):
    k = int(stdin.readline())
    n = int(stdin.readline())
    print(apartment[k][n-1])
    
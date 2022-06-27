# Baekjoon 2609

from sys import stdin
N, M = list(map(int, stdin.readline().split()))
N, M = min([N,M]), max([N,M])
mul = N*M
# 소인수분해
N_ele = {}
M_ele = {}
for i in range(2, M+1):
    if N % i == 0:
        for j in range(1, M+1):
            if N % pow(i, j) != 0:
                N_ele[i] = j-1
                N //= pow(i, j-1)
                break
    if M % i == 0:
        for j in range(1, M+1):
            if M % pow(i, j) != 0:
                M_ele[i] = j-1
                M //= pow(i, j-1)
                break
        
GCD = 1
for n in N_ele:
    if n in M_ele:
        GCD *= pow(n, min(N_ele[n], M_ele[n]))
print(GCD)
print(mul//GCD)
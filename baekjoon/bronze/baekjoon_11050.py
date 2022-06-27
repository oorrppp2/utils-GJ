# Baekjoon 11050

from sys import stdin
N, K= list(map(int, stdin.readline().split()))

def factorial(n):
    val = 1
    for i in range(1, n+1):
        val *= i
    return val
print(round(factorial(N) / (factorial(K)*(factorial(N-K)))))
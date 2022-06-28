# Baekjoon 1676

from sys import stdin
N = int(stdin.readline())

def factorial(n):
    val = 1
    for i in range(1, n+1):
        val *= i
    return val

s_facto = str(factorial(N))

for i in range(len(s_facto)-1, -1, -1):
    if int(s_facto[i]) != 0:
        print(len(s_facto) - i -1)
        exit(0)
    
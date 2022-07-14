# Baekjoon 9375

from sys import stdin
T = int(stdin.readline())

def factorial(n):
    val = 1
    for i in range(2, n+1):
        val *= i
    return val

def combi(n,m):
    return factorial(n) // (factorial(m) * factorial(n-m))

for t in range(T):
    N = int(stdin.readline())
    wear = {}
    ans = 1
    for i in range(N):
        cloth, index = list(stdin.readline().split())
        try:
            wear[index].append(cloth)
        except:
            wear[index] = [cloth]
            
    for i in wear:
        ans *= combi(len(wear[i])+1, 1)

    print(ans-1)
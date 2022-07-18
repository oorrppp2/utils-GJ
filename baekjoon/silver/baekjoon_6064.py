# Baekjoon 6064
from sys import stdin
from collections import deque

T = int(stdin.readline())

def prime_factorization(x):
    factorization = {}
    while x != 1:
        for i in range(2,x+1):
            if x % i == 0:
                try:
                    factorization[i] += 1
                except:
                    factorization[i] = 1
                x //= i
                break
    return factorization

def LCM(x, y):
    fact_x = prime_factorization(x)
    fact_y = prime_factorization(y)
    fact_set = set()
    lcm = 1
    for f in fact_x:
        fact_set.add(f)
    for f in fact_y:
        fact_set.add(f)
    for f in fact_set:
        try:
            lcm *= pow(f, max(fact_x[f], fact_y[f]))
        except:
            try:
                lcm *= pow(f, fact_x[f])
            except:
                lcm *= pow(f, fact_y[f])
    return lcm

for t in range(T):
    M, N, x, y = list(map(int, stdin.readline().split()))
    lcm_MN = LCM(M, N)
    possible_k = deque([])
    is_found = False
    if x > M or y > N:
        print(-1)
        continue

    for k in range(0, lcm_MN, M):
        possible_k.append(k + x)
    for k in range(0, lcm_MN, N):
        try:
            while possible_k[0] < k+y:
                possible_k.popleft()
            if possible_k[0] == k+y:
                print(k+y)
                is_found = True
                break
        except:
            break

    if not is_found:
        print(-1)
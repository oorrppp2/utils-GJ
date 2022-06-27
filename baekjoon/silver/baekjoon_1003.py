# Baekjoon 1003

from sys import stdin
T = int(stdin.readline())
for t in range(T):
    fibo1 = [1,0]
    fibo2 = [0,1]
    N = int(stdin.readline())
    if N == 0:
        print(fibo1[0], fibo1[1])
    elif N == 1:
        print(fibo2[0], fibo2[1])
    else:
        new_fibo = [0,0]
        for i in range(2, N+1):
            new_fibo[0] = fibo1[0] + fibo2[0]
            new_fibo[1] = fibo1[1] + fibo2[1]
            fibo1 = fibo2.copy()
            fibo2 = new_fibo.copy()
        
        print(new_fibo[0], new_fibo[1])

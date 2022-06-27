# Baekjoon 12865*
"""
    * signed problems were unsolved problems. 
"""

from sys import stdin
N, K = list(map(int, stdin.readline().split()))
l = [[0,0]]
dp = [[0]*(K+1) for _ in range(N+1)]
for i in range(N):
    W, V = list(map(int, stdin.readline().split()))
    l.append([W,V])

for i in range(1,N+1):
    w, v = l[i]
    for j in range(1,K+1):
        if j < w:
            dp[i][j] = dp[i-1][j]
        else:
            dp[i][j] = max(dp[i-1][j], dp[i-1][j-w]+v)

print(dp[N][K])

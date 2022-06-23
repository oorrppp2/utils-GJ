import sys

K, P, N = map(int, input().split())
ans = K
for i in range(N):
    ans = (ans * P) % 1000000007

print(ans)
# 2 + 2*3 + 2*3*3
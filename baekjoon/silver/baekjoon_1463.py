# Baekjoon 1463

from sys import stdin
N = int(stdin.readline())
ans = 0
arr = [0 for _ in range(N+1)]

for i in range(2, N+1):
    if i % 6 == 0:
        arr[i] = min(arr[i//3], arr[i//2], arr[i-1])+1
    elif i % 3 == 0:
        arr[i] = min(arr[i//3], arr[i-1])+1
    elif i % 2 == 0:
        arr[i] = min(arr[i//2], arr[i-1])+1
    else:
        arr[i] = arr[i-1]+1

print(arr[N])
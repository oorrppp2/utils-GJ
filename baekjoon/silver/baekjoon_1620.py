# Baekjoon 1620

from sys import stdin

N, M = list(map(int, stdin.readline().split()))
pocket_dict = {}
for i in range(N):
    s = stdin.readline().split()[0]
    pocket_dict[s] = str(i+1)
    pocket_dict[str(i+1)] = s
    
for i in range(M):
    s = stdin.readline().split()[0]
    print(pocket_dict[s])
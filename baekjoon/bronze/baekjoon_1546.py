# Baekjoon 1546

from sys import stdin
N = int(stdin.readline())
scores = list(map(int, stdin.readline().split()))
best_score = max(scores)
for i in range(N):
    scores[i] = scores[i] / best_score * 100
print(sum(scores) / N)
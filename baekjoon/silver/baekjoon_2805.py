# Baekjoon 2805

from sys import stdin
N, M = list(map(int, stdin.readline().split()))
trees = list(map(int, stdin.readline().split()))

# 이분탐색
start = 0
end = max(trees)

while True:
    mid = (start + end) // 2
    if start == end-1:
        break
    cut_trees = 0
    for tree in trees:
        if tree > mid:
            cut_trees += tree-mid
    if cut_trees < M:
        end = mid
    else:
        start = mid
    
print(mid)
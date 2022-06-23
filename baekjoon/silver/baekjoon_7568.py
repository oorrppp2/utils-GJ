# Baekjoon 7568

N = int(input())
heights = []
weights = []
ans = []
for i in range(N):
    height, weight = input().split()
    heights.append(int(height))
    weights.append(int(weight))

for i in range(len(heights)):
    height = heights[i]
    weight = weights[i]
    level = 1
    for j in range(len(heights)):
        if heights[j] > height and weights[j] > weight:
            level += 1
    ans.append(level)

for a in ans:
    print(a, end=' ')
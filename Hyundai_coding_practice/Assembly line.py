import sys

N = int(input())
lA = []
lB = []
lAtoB = []
lBtoA = []
for i in range(N-1):
    A, B, AtoB, BtoA = map(int, input().split())
    lA.append(A)
    lB.append(B)
    lAtoB.append(AtoB)
    lBtoA.append(BtoA)
A, B = map(int, input().split())
lA.append(A)
lB.append(B)

# cost = 현재까지 온 거리 + 남은 거리
current_line = ""
C = []
index = 0
ans = 0
# C[x] = [cost, ans, index, current_line]
C.append([sum(lA[index:]), lA[0], index, "A"])
C.append([sum(lA[index:]), lB[0], index, "B"])

while 1:
    node = min(C)
    # print(node)
    ans = node[1]
    index = node[2] + 1
    current_line = node[3]
    if index == N:
        break
    else:
        if current_line == "A":
            C.append([ans + lA[index] + sum(lA[index:]), ans + lA[index], index, "A"])
            C.append([ans + lAtoB[index-1] + lB[index] + sum(lB[index:]), ans + lAtoB[index-1] + lB[index], index, "B"])
        else:
            C.append([ans + lB[index] + sum(lB[index:]), ans + lB[index], index, "B"])
            C.append([ans + lBtoA[index-1] + lA[index] + sum(lA[index:]), ans + lBtoA[index-1] + lA[index], index, "A"])
    C.remove(node)

ans = 0
for i in range(1, len(C)):
    node = C[i]
    if node[2] == N-1 and ans == 0:
        ans = C[i][1]
    elif node[2] == N-1 and node[1] < ans:
        ans = C[i][1]
print(ans)
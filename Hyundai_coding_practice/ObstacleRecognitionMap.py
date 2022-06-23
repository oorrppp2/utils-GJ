import sys

N = int(input())
M = []
B = []
Mask = [[0 for i in range(N)] for j in range(N)]
MaskI = 1
for i in range(N):
    L = sys.stdin.readline().strip()
    l = []
    for j in range(N):
      l.append(int(L[j]))
    M.append(l)

for n in range(N):
    for m in range(N):
        if M[n][m] == 1 and Mask[n][m] == 0:
            Mask[n][m] = MaskI
            WorkingQ = [[n,m]]
            blocks = 1
            while len(WorkingQ) != 0:
                i, j = WorkingQ.pop()
                if i-1 >= 0 and M[i-1][j] == 1 and Mask[i-1][j] == 0:
                    Mask[i-1][j] = MaskI
                    WorkingQ.append([i-1,j])
                    blocks += 1
                if j-1 >= 0 and M[i][j-1] == 1 and Mask[i][j-1] == 0:
                    Mask[i][j-1] = MaskI
                    WorkingQ.append([i,j-1])
                    blocks += 1
                if i+1 < N and M[i+1][j] == 1 and Mask[i+1][j] == 0:
                    Mask[i+1][j] = MaskI
                    WorkingQ.append([i+1,j])
                    blocks += 1
                if j+1 < N and M[i][j+1] == 1 and Mask[i][j+1] == 0:
                    Mask[i][j+1] = MaskI
                    WorkingQ.append([i,j+1])
                    blocks += 1
            MaskI += 1
            B.append(blocks)
B.sort()
print(MaskI-1)
for i in range(MaskI-1):
    print(B[i])
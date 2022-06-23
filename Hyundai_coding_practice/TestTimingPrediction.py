import sys

N, M = map(int, input().split())
Map = []
for i in range(N):
    r = list(map(int, sys.stdin.readline().split()))
    Map.append(r)

WorkingQ = [[0,0]]
# print(sum(M))
# while max(M) != 0:
while len(WorkingQ) != 0:
    n, m = WorkingQ.pop()
    if n-1 >= 0 and Map[n-1][m] == 0:
        Map[n-1][m] = -1
        WorkingQ.append([n-1,m])
    if m-1 >= 0 and Map[n][m-1] == 0:
        Map[n][m-1] = -1
        WorkingQ.append([n,m-1])
    if n+1 < N and Map[n+1][m] == 0:
        Map[n+1][m] = -1
        WorkingQ.append([n+1,m])
    if m+1 < M and Map[n][m+1] == 0:
        Map[n][m+1] = -1
        WorkingQ.append([n,m+1])

list_sum = 0
ans = 0
while list_sum != -((N-2)*(M-2)):
    ans += 1
# for k in range(4):
#     print("list_sum : ", list_sum)
    list_sum = 0
    for i in range(1, N-1):
        for j in range(1, M-1):
            if Map[i][j] == 1:
                if (Map[i-1][j] == -1) + (Map[i][j-1] == -1) + (Map[i+1][j] == -1) + (Map[i][j+1] == -1) >= 2:
                    Map[i][j] = 2

    for i in range(1, N - 1):
        for j in range(1, M - 1):
            if Map[i][j] == 2:
                Map[i][j] = -1
                if Map[i-1][j] == 0 or Map[i][j-1] == 0 or Map[i+1][j] == 0 or Map[i][j+1] == 0:
                    WorkingQ.append([i,j])
                    while len(WorkingQ) != 0:
                        n, m = WorkingQ.pop()
                        if n-1 >= 0 and Map[n-1][m] == 0:
                            Map[n-1][m] = -1
                            WorkingQ.append([n-1,m])
                        if m-1 >= 0 and Map[n][m-1] == 0:
                            Map[n][m-1] = -1
                            WorkingQ.append([n,m-1])
                        if n+1 < N and Map[n+1][m] == 0:
                            Map[n+1][m] = -1
                            WorkingQ.append([n+1,m])
                        if m+1 < M and Map[n][m+1] == 0:
                            Map[n][m+1] = -1
                            WorkingQ.append([n,m+1])
            list_sum += Map[i][j]

    for i in range(N):
        for j in range(M):
            print(Map[i][j], end='\t')
        print()
    print("-"*21)

print(ans)
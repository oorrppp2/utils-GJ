# Baekjoon 1012

from sys import stdin

T = int(stdin.readline())
for t in range(T):
    M, N, K = list(map(int, stdin.readline().split()))
    ground = [[0 for _ in range(M)] for _ in range(N)]
    for k in range(K):
        x, y = list(map(int, stdin.readline().split()))
        ground[y][x] = 1

    num = 2
    for i in range(N):
        for j in range(M):
            if ground[i][j] == 1:
                val = []
                if j > 0:
                    if ground[i][j-1] > 1:
                        val.append(ground[i][j-1])
                if i > 0:
                    if ground[i-1][j] > 1:
                        val.append(ground[i-1][j])

                if len(val) == 0:
                    ground[i][j] = num
                elif len(val) == 1:
                    ground[i][j] = val[0]
                elif len(val) == 2:
                    for ii in range(N):
                        for jj in range(M):
                            if ground[ii][jj] == val[0]:
                                ground[ii][jj] = val[1]
                    ground[i][j] = val[1]
                    # num = val[1]
            num += 1
        if i > 3:
            for k in range(N):
                print(ground[k])
            print("num: ", num)
            print("=============================================")

    for i in range(N):
        print(ground[i])
    ans = set()
    for i in ground:
        for j in i:
            if j != 0:
                ans.add(j)
    print(len(ans))
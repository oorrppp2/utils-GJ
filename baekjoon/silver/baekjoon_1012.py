# Baekjoon 1012

from sys import stdin

T = int(stdin.readline())
for t in range(T):
    M, N, K = list(map(int, stdin.readline().split()))
    ground = [[0 for _ in range(M)] for _ in range(N)]
    for k in range(K):
        x, y = list(map(int, stdin.readline().split()))
        ground[y][x] = 1

    # for i in range(N):
    #     print(ground[i])
    # print("="*50)
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
                    # ground[ground==val[0]] = val[1]
                    ground[i][j] = val[1]
                    num = val[1]
            num += 1
            # else:
            #     num += 1
        # print(i, ": ", ground[i])

    # for i in range(N):
    #     print(ground[i])
    # for i in range(N):
    #     print(i, ":", ground[i])
    # flattend = []
    ans = set()
    for i in ground:
        # print("i: ", i)
        for j in i:
            if j != 0:
                ans.add(j)
    #     flattend.extend([i])
    # print(flattend)
        # for j in range(M):
            # ans.add(ground[i][j])
    # print(ans)
    print("ans: ", len(ans))
# Baekjoon 2667

from sys import stdin
N = int(stdin.readline())

village = []
numbered_village = [[0 for _ in range(N)] for _ in range(N)]
for i in range(N):
    village.append(stdin.readline().split()[0])

dange_num = 2
for i in range(N):
    for j in range(N):
        if village[i][j] == '1':
            num1, num2 = 0, 0
            if j > 0:
                if numbered_village[i][j-1] != 0:
                    num1 = numbered_village[i][j-1]
            if i > 0:
                if numbered_village[i-1][j] != 0:
                    num2 = numbered_village[i-1][j]

            if num1 == 0 and num2 == 0:
                numbered_village[i][j] = dange_num
            elif num1 == num2:
                numbered_village[i][j] = num1
            else:
                if num1 * num2 != 0:
                    for ii in range(N):
                        for jj in range(N):
                            if numbered_village[ii][jj] == num2:
                                numbered_village[ii][jj] = num1
                    numbered_village[i][j] = num1
                else:
                    numbered_village[i][j] = max(num1, num2)
        dange_num += 1

result = {}
for i in range(N):
    for j in range(N):
        num = numbered_village[i][j]
        if num != 0:
            try:
                result[num] += 1
            except:
                result[num] = 1

result_list = []
for r in result:
    result_list.append(result[r])
result_list = sorted(result_list)
print(len(result_list))
for ans in result_list:
    print(ans)

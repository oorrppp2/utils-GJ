# Baekjoon 1181

from sys import stdin
N = int(stdin.readline())

ans = [[0, '']]
for i in range(N):
    s = stdin.readline().split()[0]
    exist = False
    # for s_ in ans:
    #     if s == s_[1]:
    #         exist = True
    #         break
    # if not exist:
    ans.append([len(s), s])
# set(ans)
ans = sorted(ans)
# print(ans)
for i in range(1, len(ans)):
    if ans[i][1] != ans[i-1][1]:
        print(ans[i][1])
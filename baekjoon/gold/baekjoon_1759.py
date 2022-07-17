# Baekjoon 1759

from sys import stdin

L, C = list(map(int, stdin.readline().split()))
alphabet = list(stdin.readline().split())
alphabet.sort()
ans = alphabet.copy()
ans_index = [[i] for i in range(C)]

while len(ans[0]) < L:
    s = ans.pop(0)
    index = ans_index.pop(0)
    last_index = index[-1]
    for i in range(last_index+1, C):
        ans.append(s+alphabet[i])
        ans_index.append(index+[i])


for s in ans:
    zaum_cnt = 0
    moum_cnt = 0
    for i in range(len(s)):
        if s[i] in ['a', 'e', 'i', 'o', 'u']:
            moum_cnt += 1
        else:
            zaum_cnt += 1
        if moum_cnt > 0 and zaum_cnt > 1:
            print(s)
            break

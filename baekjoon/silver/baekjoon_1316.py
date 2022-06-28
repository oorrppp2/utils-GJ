# Baekjoon 1316

from sys import stdin

N = int(stdin.readline())

ans = 0
for t in range(N):
    s = stdin.readline().split()[0]
    s_dict = {}
    is_groupword = True
    for i in range(len(s)):
        if i == 0:
            s_dict[s[i]] = True
        else:
            try:
                if s[i] != s[i-1]:
                    if s_dict[s[i]]:
                        is_groupword = False
                        break
            except:
                s_dict[s[i]] = True
    if is_groupword:
        ans += 1

print(ans)    
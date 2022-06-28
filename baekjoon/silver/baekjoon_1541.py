# Baekjoon 1541

from sys import stdin

s = stdin.readline().split()[0]
ans = 0
num = ''
sub_num = 0
is_substracting = False
for i in range(len(s)):
    if s[i] == '+':
        if is_substracting:
            ans -= int(num)
        else:
            ans += int(num)
        num = ''
    elif s[i] == '-':
        if is_substracting:
            ans -= int(num)
        else:
            is_substracting = True
            ans += int(num)
        num = ''
    else:
        num += s[i]
if is_substracting:
    ans -= int(num)
else:
    ans += int(num)

print(ans)
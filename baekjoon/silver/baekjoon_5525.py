# Baekjoon 5525

from sys import stdin

N = int(stdin.readline())
M = int(stdin.readline())
ans = 0
PN = 'I'
for i in range(N):
    PN += 'OI'

s = stdin.readline().split()[0]

i = 0
while i < M-(N*2):
    print('i: ', i)
    if s[i:i+(N*2)+1] == PN:
        ans += 1
        for j in range(i+(N*2)+1, M-(N*2)):
            if s[j:j+2] == "OI":
                ans += 1
                i += 2
            else:
                i += 1
                break
    else:
        i += 1
        no_exist = True
        for j in range(i, M-(N*2)):
            print(s[j:j+3], j)
            if s[j:j+3] == 'IOI':
                i = j
                no_exist = False
                break
        if no_exist:
            break
print(ans)
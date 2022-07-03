# Baekjoon 1058

from sys import stdin

N = int(stdin.readline())
relationship = {}
ans = 0
for i in range(N):
    friend = stdin.readline().rstrip()
    relationship[i] = []
    for f in range(len(friend)):
        if friend[f] == 'Y':
            relationship[i].append(f)


for i in range(N):
    friend_2 = relationship[i].copy()
    friends = []
    for f in friend_2:
        friends += relationship[f].copy()
    friend_2 += friends
    friend_2 = list(set(friend_2))
    for f in friend_2:
        if f == i:
            friend_2.remove(f)
    ans = len(friend_2) if len(friend_2) > ans else ans

print(ans)
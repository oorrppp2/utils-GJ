# Baekjoon 1874

from sys import stdin
N = int(stdin.readline())
progression = []
for i in range(N):
    progression.append(int(stdin.readline()))

start = 1
stack = []
ans = []
for i in range(N):
    stack.append(i+1)
    ans.append("+")
    while stack[-1] == progression[0]:
        stack.pop()
        progression.pop(0)
        ans.append("-")
        if len(stack) == 0 or len(progression) == 0:
            break

if len(progression) == 0:
    for s in ans:
        print(s)
else:
    print("NO")

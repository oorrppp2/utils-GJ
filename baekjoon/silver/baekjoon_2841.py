# Baekjoon 2841

from sys import stdin

N, M = list(map(int, stdin.readline().split()))

ans = 0
lines = [[] for _ in range(6)]

for i in range(N):
    line, fret = list(map(int, stdin.readline().split()))
    line -= 1
    if len(lines[line]) > 0:
        if lines[line][-1] < fret:
            lines[line].append(fret)
            ans += 1
        else:
            while len(lines[line]) > 0:
                if lines[line][-1] <= fret:
                    break
                lines[line].pop()
                ans += 1
            if len(lines[line]) == 0:
                lines[line].append(fret) 
                ans += 1
            elif lines[line][-1] < fret:
                lines[line].append(fret) 
                ans += 1
    else:
        lines[line].append(fret) 
        ans += 1
print(ans)
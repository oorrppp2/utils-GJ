# Baekjoon 1244

from sys import stdin

N = int(stdin.readline())
switch = list(map(int, stdin.readline().split()))

S = int(stdin.readline())
for s in range(S):
    student, number = list(map(int, stdin.readline().split()))
    if student == 1:
        for i in range(len(switch)):
            if (i+1) % number == 0:
                switch[i] = (switch[i] + 1) % 2
    else:
        i = 0
        while number - 1 - i >= 0 and number - 1 + i < N:
            if switch[number - 1 - i] == switch[number - 1 + i]:
                switch[number - 1 - i], switch[number - 1 + i] = (switch[number - 1 - i] + 1) % 2, (switch[number - 1 + i] + 1) % 2
            else:
                break
            i += 1

for i in range(1, len(switch)+1):
    print(switch[i-1], end=' ')
    if i % 20 == 0:
        print()
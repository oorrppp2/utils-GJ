# Baekjoon 1764

from sys import stdin

N, M = list(map(int, stdin.readline().split()))

name_dict = {}
jobnom = []
for i in range(N):
    name = stdin.readline().split()[0]
    name_dict[name] = True
for i in range(M):
    name = stdin.readline().split()[0]
    try:
        if name_dict[name]:
            jobnom.append(name)
    except:
        continue

jobnom = sorted(jobnom)
print(len(jobnom))
for nom in jobnom:
    print(nom)
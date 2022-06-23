# Baekjoon 10814

from sys import stdin
N = int(stdin.readline())
name_list = {}
for i in range(1, 201):
    name_list[i] = []
for i in range(N):
    age, name = stdin.readline().split()
    name_list[int(age)].append(name)

for i in range(1, 201):
    for name in name_list[i]:
        print(str(i) + ' ' + name)
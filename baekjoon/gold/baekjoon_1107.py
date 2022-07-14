# Baekjoon 1107

from sys import stdin

channel = stdin.readline()
channel_i = int(channel)
n = int(stdin.readline())
choosable = [i for i in range(10)]
possible_num = -1
if n > 0:
    broken = list(map(int, stdin.readline().split()))
    for key in broken:
        choosable.remove(key)

if len(choosable) == 0:
    print(abs(channel_i - 100))
    exit(0)
for i in range(500000):
    if channel_i - i >= 0:
        num = str(channel_i - i)
        is_possible = True
        for n in num:
            # print(n)
            if int(n) not in choosable:
                is_possible = False
        
        if is_possible:
            possible_num = int(num)
            break
    num = str(channel_i + i)
    is_possible = True
    for n in num:
        if int(n) not in choosable:
            is_possible = False
    
    if is_possible:
        possible_num = int(num)
        break
    

ans = abs(channel_i - possible_num) + len(str(possible_num))
if ans < abs(channel_i - 100):
    print(ans)
else:
    print(abs(channel_i - 100))

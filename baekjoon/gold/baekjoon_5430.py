# Baekjoon 5430

from sys import stdin
T = int(stdin.readline())

for t in range(T):
    command = stdin.readline().split()[0]
    N = int(stdin.readline())
    initial_list = stdin.readline()[1:-2]
    if len(initial_list) != 0:
        initial_list = initial_list.split(',')
    pop_front = True
    error = False
    for c in command:
        if c == 'R':
            pop_front = not pop_front
        elif c == 'D':
            if len(initial_list) == 0:
                error = True
                break
            if pop_front:
                initial_list.pop(0)
            else:
                initial_list.pop()
    if error:
        print('error')
        continue
    
    if len(initial_list) == 0:
        print('[]')
    else:
        ans = '['
        if not pop_front:
            for i in range(len(initial_list)):
                ans += initial_list.pop()
                ans += ','
        else:
            for c in initial_list:
                ans += c
                ans += ','
            
        ans = ans[:-1] + ']'
        print(ans)

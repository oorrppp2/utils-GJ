# Baekjoon 9019

from sys import stdin
from collections import deque

T = int(stdin.readline())

for t in range(T):
    A, B = list(map(int, stdin.readline().split()))
    
    visited = [False for _ in range(10000)]
    working = deque()
    working.append([A, ''])
    visited[A] = True
    while len(working) > 0:
        current = working.popleft()
        num = current[0]
        command = current[1]

        # D
        next_num = (num*2) % 10000
        # print("D: ", next_num)
        if not visited[next_num]:
            if next_num == B:
                print(command + 'D')
                break
            working.append([next_num, command+'D'])
            visited[next_num] = True

        # S
        next_num = num - 1 if num-1 >= 0 else 9999
        # print("S: ", next_num)
        if not visited[next_num]:
            if next_num == B:
                print(command + 'S')
                break
            working.append([next_num, command+'S'])
            visited[next_num] = True

        # L
        next_num = (num // 1000) + (num % 1000) * 10
        next_num = int(('%04d' % num)[1:]+('%04d' % num)[0])
        # print("L: ", next_num)
        if not visited[next_num]:
            if next_num == B:
                print(command + 'L')
                break
            working.append([next_num, command+'L'])
            visited[next_num] = True

        # R
        next_num = (1000 * (num % 10)) + (num // 10)
        # print("R: ", next_num)
        if not visited[next_num]:
            if next_num == B:
                print(command + 'R')
                break
            working.append([next_num, command+'R'])
            visited[next_num] = True


# Baekjoon 9019

# import time
# for i in range(10000):
#     for j in range(10000):
#         if i == j:
#             continue
#         t = time.time()
#         A, B = i, j
        
#         visited = [False for _ in range(10000)]
#         working = deque()
#         working.append([A, ''])
#         visited[A] = True
#         while len(working) > 0:
#             current = working.popleft()
#             num = current[0]
#             command = current[1]

#             # D
#             next_num = (num*2) % 10000
#             # print("D: ", next_num)
#             if not visited[next_num]:
#                 if next_num == B:
#                     print(command + 'D')
#                     break
#                 working.append([next_num, command+'D'])
#                 visited[next_num] = True

#             # S
#             next_num = num - 1 if num-1 >= 0 else 9999
#             # print("S: ", next_num)
#             if not visited[next_num]:
#                 if next_num == B:
#                     print(command + 'S')
#                     break
#                 working.append([next_num, command+'S'])
#                 visited[next_num] = True

#             # L
#             next_num = (num // 1000) + (num % 1000) * 10
#             next_num = int(('%04d' % num)[1:]+('%04d' % num)[0])
#             # print("L: ", next_num)
#             if not visited[next_num]:
#                 if next_num == B:
#                     print(command + 'L')
#                     break
#                 working.append([next_num, command+'L'])
#                 visited[next_num] = True

#             # R
#             next_num = (1000 * (num % 10)) + (num // 10)
#             # print("R: ", next_num)
#             if not visited[next_num]:
#                 if next_num == B:
#                     print(command + 'R')
#                     break
#                 working.append([next_num, command+'R'])
#                 visited[next_num] = True

#         print(i, j, 'time: ', time.time()-t)

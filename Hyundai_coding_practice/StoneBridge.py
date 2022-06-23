import sys
import random
import time
start = time.time()

# N = int(input())
# S = list(map(int, (sys.stdin.readline().split())))
# count = 0
# count_list = []
# mode = "up"
# for i in range(N):
#     if mode == "up" and S[i] - S[i-1] > 0:
#         count += 1
#         if i+1 != N and S[i+1] < S[i]:
#             mode = "down"
#     elif mode == "down" and S[i] - S[i-1] < 0:
#         count += 1
#         if i+1 != N and S[i+1] > S[i]:
#             mode = "up"
#             count_list.append(count)
#             count = 0
# count_list.append(count+1)
# print(max(count_list))


N = int(input())
S = list(map(int, (sys.stdin.readline().split())))
top = max(S)
count = 0
mode = "up"
ref = N-1
for i in range(N-1, 0, -1):
    if S[i] != top:
        if S[ref] - S[i-1] < 0:
            ref = i
            count += 1
    else:
        ref = i
        if S[ref] - S[i-1] < 0:
            ref = i
            count += 1

    if mode == "up" and S[i] - S[i-1] > 0:
        count += 1
        if i+1 != N and S[i+1] < S[i]:
            mode = "down"
    elif mode == "down" and S[i] - S[i-1] < 0:
        count += 1
        if i+1 != N and S[i+1] > S[i]:
            mode = "up"
            count_list.append(count)
            count = 0
count_list.append(count+1)
print(max(count_list))
for i in range(5, 0, -1):
    print(i)


A = random.sample(range(100000000), 300000)
print(max(A))
print("time :", time.time() - start)
# Baekjoon 17626

from sys import stdin

N = int(stdin.readline())

square = int(N ** 0.5)

ans = 5
import time
t = time.time()

for i in range(square, 0, -1):
    count = 1
    square_sum1 = pow(i, 2)
    # square_sum1 = i ** 2
    if square_sum1 == N:
        if count < ans:
            ans = count
        # print(count)
        # exit(0)
    res_square = int((N - square_sum1) ** 0.5)
    # res_square = int(math.sqrt(N - square_sum1))
    for j in range(res_square, 0, -1):
        count = 2
        if count >= ans:
            break
        c_square = square_sum1 + pow(j, 2)
        if c_square > N:
            continue
        elif c_square == N:
            if count < ans:
                ans = count
            # print(count)
            # exit(0)
        square_sum2 = c_square
        res_square = int((N - square_sum2) ** 0.5)
        # res_square = int(math.sqrt(N - square_sum2))
        for k in range(res_square, 0, -1):
            count = 3
            if count >= ans:
                break
            c_square = square_sum2 + pow(k, 2)
            # print("k: ", k)
            # print("square_sum1: ", square_sum1)
            # print("square_sum2: ", square_sum2)
            # print("c_square: ", c_square)
            if c_square > N:
                continue
            elif c_square == N:
                if count < ans:
                    ans = count
                # print(count)
                # exit(0)
            square_sum3 = c_square
            res_square = int((N - square_sum3) ** 0.5)
            # res_square = int(math.sqrt(N - square_sum3))
            # for l in range(res_square, 0, -1):
            #     # print("square_sum1: ", square_sum1)
            #     # print("square_sum2: ", square_sum2)
            #     # print("square_sum3: ", square_sum3)
            #     count = 4
            #     if count >= ans:
            #         break
            #     c_square = square_sum3 + pow(l, 2)
            #     if c_square > N:
            #         continue
            #     elif c_square == N:
            #         if count < ans:
            #             ans = count
                    # print(count)
                    # exit(0)
if ans == 5:
    ans = 4
print(ans)

print(time.time() - t)


# # import sys
# # input = sys.stdin.readline
# # n=int(input())
# # d=[0]*50001
# # import time
# # t = time.time()
# # for i in range(1,n+1):
# #   li=[]
# #   j=1
# #   while i>=j**2:
# #     li.append(d[i-j**2])
# #     j+=1
# #   d[i]=min(li)+1
# # print(d[n])
# # print(time.time() - t)

# Baekjoon 17626

# from sys import stdin

# N = int(stdin.readline())

# arr = [0 for _ in range(50001)]
# square = int(N ** 0.5)

# for i in range(1, N+1):
#     candidates = []
#     for j in range(1, int(i ** 0.5)+1):
#         candidates.append(arr[i-j**2])
#     arr[i] = min(candidates)+1

# print(arr[N])
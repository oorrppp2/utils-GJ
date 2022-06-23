# import numpy as np

# a = np.array([2, -1, 4])
# b = np.array([-3, 2, 5])

# print(np.sqrt(pow((a[0]-b[0]), 2) +
#             pow((a[1]-b[1]), 2) +
#             pow((a[2]-b[2]), 2)))

# print(np.linalg.norm(a-b))

# aList = [2,4,6,8,5,3,1]
# min1=1000000000 
# min2=1000000000 
# for i in range(len(aList)):
#     if min2>aList[i]:
#         min2=aList[i]
#         if min1 > min2:
#             min1,min2 = min2, min1
# print(min1)
# print(min2)


# aList = [2,4,6,8,5,3,1]
# aList = [3,4,5,7,8,9, 2000 ,1,11,50,60,70,80,90,100,1000]

# max1 = 0
# max2 = 0
# for i in range(len(aList)):
#     if max2<aList[i]:
#         max2 = aList[i]
#         if max1 < max2:
#             max1,max2 = max2, max1
# print(max1)
# print(max2)

# n = int(input("n"))
# f = []
# for i in range(n):
#     if i == 0:
#         f.append(1)
#     elif i == 1:
#         f.append(1)
#     else:
#         f.append(f[i-1]+f[i-2])
# print('f: ', f)
# print("Fibonacci sum: ", sum(f))

# print(len(s))
# tmp = s[3:]
# print(tmp)
# print(s[:1])
# print(s[:1]+tmp)

# def solution(s):
#     i = 0
#     tmp = []
#     for i in range(len(s)):
#         if len(tmp) != 0 and tmp[-1] == s[i]:
#             tmp.pop()
#         else:
#             tmp.append(s[i])
#     return 1 if len(tmp) == 0 else 0
# s = 'baabaa'
# print(s)
# s.join('')
# print(s)
# print(solution(s))

# def solution(n, m):
#     num1, num2 = (n, m) if n < m else (m, n)
#     GCD = 1
#     LCM = 1
#     CM_1 = []
#     CM_2 = []

#     for i in range(1, num2):
#         if num1 % i == 0 and num2 % i == 0:
#             GCD = i
#     for i in range(1, num1+1):
#         if num2*i != 1:
#             CM_1.append(num2*i)
#     for i in range(1, num2+1):
#         if num1*i != 1:
#             CM_2.append(num1*i)
#     for i in CM_1:
#         if i in CM_2:
#             LCM = i
#             return [GCD, LCM]
#     return [GCD, 1]
            
    
# def solution(n):
#     result = 0
#     binary = bin(n)
#     print(binary)
#     print(type(binary))
#     return result

# print(solution(8))

# N, M = list(map(int, input().split()))
# K = list(map(int, input().split()))
# answer = 0
# for i in range(1, N+1):
#     for j in K:
#         if i % j == 0:
#             answer += i
#             break

# print(answer)

# def solution(s):
#     tmp = []
#     for i in range(len(s)):
#         if len(tmp) != 0 and tmp[-1] == s[i]:
#             tmp.pop()
#         else:
#             tmp.append(s[i])
#     return 1 if len(tmp) == 0 else 0

# s = 'abbss'
# print(solution(s))

def solution(lines):
    answer = 0
    return answer
# 2
# myList = [11, 22, 23, 99, 81, 93, 35]
# mySum = 0
# for ele in myList:
#     mySum += ele
# print("정수들의 합은 ", mySum)

# 5
# nInput = int(input("정수를 입력하시오: "))
# for i in range(1, nInput+1):
#     for j in range(1, i+1):
#         print(j, end=' ')
#     print('')

# 10
# n = int(input("n의 값을 입력하시오: "))
# result = 0
# for i in range(1, n+1):
#     result += i ** 2
# print("계산값은 ", result, "입니다.")

# 13
# n = int(input("몇 번째 항까지 구할까요? "))
# f = [0, 1]
# for i in range(n-1):
#     f.append(f[i] + f[i+1])
# print(f)

# 17


# for i in range(20):
#     if i % 3 == 0 and i % 5 == 0:
#         print("fizzbuzz")
#     elif i % 3 == 0:
#         print("fizz")
#     elif i % 5 == 0:
#         print("buzz")
#     else:
#         print("*")


# for i in range(10):
i = 0
while i < 10 :
    i += 1
    print(i)
    if i == 3:
        i += 1
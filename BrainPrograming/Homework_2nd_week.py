
# 1
# nFirstNum = int(input("정수를 입력하시오: "))
# nSecondNum = int(input("정수를 입력하시오: "))
#
# print("약수입니다." if nFirstNum % nSecondNum == 0 else "약수가 아닙니다.")


# 4
# import numpy as np
# dRadius = float(input("원의 반지름: "))
# print("원의 면적: " + str(np.pi * dRadius ** 2) if dRadius >= 0 else "잘못된 값입니다")

# 6
import random
nUsersChoice = int(input("선택하시오(1: 가위 2: 바위 3: 보) "))
nComputersChoice = random.randint(1,3)
print("컴퓨터의 선택(1: 가위 2: 바위 3: 보)", nComputersChoice)
if nUsersChoice == nComputersChoice:
    print("비겼음")
elif (nUsersChoice == 1 and nComputersChoice == 2) or \
        (nUsersChoice == 2 and nComputersChoice == 3) or \
        (nUsersChoice == 3 and nComputersChoice == 1):
    print("컴퓨터가 이겼음")
else:
    print("사용자가 이겼음")


# 14

# """
#     If get_discriminant(a, b, c) > 0 : real root
#     If get_discriminant(a, b, c) = 0 : multiple root
#     If get_discriminant(a, b, c) < 0 : imaginary root
# """
# import numpy as np
# def get_discriminant(a, b, c):
#     return b ** 2 - 4 * a * c
# a = float(input("a를 입력하시오: "))
# b = float(input("b를 입력하시오: "))
# c = float(input("c를 입력하시오: "))
#
# if get_discriminant(a, b, c) == 0:
#     print("실근은 ", (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a), "입니다.")
# elif get_discriminant(a, b, c) > 0:
#     print("실근은 ", (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a), "과 ", (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a), "입니다.")
# else:
#     print("실근이 존재하지 않습니다.")
#

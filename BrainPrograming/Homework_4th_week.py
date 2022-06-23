# 1, 3, 8, 11, 14

# # 1
# import numpy as np
# def get_peri(r=5.0):
#     return 2.0 * np.pi * r
# print(get_peri())
# print(get_peri(4.0))

# # 3
# def calc(n1, n2):
#     sum = n1 + n2
#     sub = n1 - n2
#     mul = n1 * n2
#     div = n1 / n2
#     return sum, sub, mul, div
# nInput1 = int(input("첫 번째 정수를 입력하시오: "))
# nInput2 = int(input("두 번째 정수를 입력하시오: "))
# while nInput2 == 0:
#     nInput2 = int(input("0이 아닌 두 번째 정수를 입력하시오: "))
# sum, sub, mul, div = calc(nInput1, nInput2)
# print("({0} + {1} ) = {2}".format(nInput1, nInput2, sum))
# print("({0} - {1} ) = {2}".format(nInput1, nInput2, sub))
# print("({0} * {1} ) = {2}".format(nInput1, nInput2, mul))
# print("({0} / {1} ) = {2}".format(nInput1, nInput2, div))

# # 8
# def getMoneyText(amount):
#     digit_list = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
#     cText = ""
#     if amount == 0:
#         cText = "영 원"
#     elif amount < 10:
#         cText = digit_list[int(amount % 10)] + "원"
#     elif amount < 100:
#         cText = digit_list[int((amount/10) % 10)] + "십 " + digit_list[int(amount % 10)] + "원"
#     elif amount < 1000:
#         if int((amount/10)%10) == 0:
#             cText = digit_list[int((amount/100) % 10)] + "백 " + digit_list[int(amount % 10)] + "원"
#         else:
#             cText = digit_list[int((amount/100) % 10)] + "백 " + digit_list[int((amount/10) % 10)] + "십 " + digit_list[int(amount % 10)] + "원"
#     elif amount == 1000:
#         cText = "천 원"
#     return cText
# nInput = int(input("1000 이하의 금액을 입력하시오: "))
# print(getMoneyText(nInput))

# # 11
# def deci2bin(n):
#     cBinary = ""
#     while n != 0:
#         cBinary = str(n%2) + cBinary
#         n = int(n/2)
#     return cBinary
# nInput = int(input("10진수: "))
# print(deci2bin(nInput))

# 14
def test(n, pred):
    result = n / pred
    return result if abs(pred - result) < 0.000001 else test(n, (pred+result)/2)
def sqrt(n):
    pred = 1
    return test(n, pred)
dInput = float(input("제곱근을 구할 수를 입력하세요: "))
print(sqrt(dInput))


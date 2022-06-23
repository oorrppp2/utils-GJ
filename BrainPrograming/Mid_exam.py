# def fun(n=[]):
#     non_nega_nums = 0
#     nega_nums = 0
#     for n in nList:
#         if n < 0:
#             nega_nums += 1
#         else:
#             non_nega_nums += 1
#     return non_nega_nums, nega_nums
#
# print(fun(1,2,-1))

# print(i for i in range(-100, 100))

def fun(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    s = s1 & s2
    return ''.join(s)



print(fun("abc", "bcd"))
# 4, 5, 14, 16

# # 4
# lInitialList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# lGoalList = [-i if 3 <= i <= 8 else i for i in lInitialList]
# print("실행전", lInitialList)
# print("실행후", lGoalList)

# # 5
# lStrList = ['aba', 'xyz', 'abc', '121']
# print("문자열의 개수=", sum([c[0] == c[-1] for c in lStrList]))

# # 14
# import numpy.random as random
# lInitialMap = [[1] * 10] * 10
# lInitialMap = [[i * random.randint(100)*0.01 for i in row] for row in lInitialMap]
# lBombMap = [['#' if i < 0.3 else '.' for i in row] for row in lInitialMap]
# for row in lBombMap:
#     print(row)

# 16
lDecList = [i for i in range(2, 101)]
lDelList = [i for j in lDecList for i in lDecList if i % j == 0 and i > j]
lDecList = [i for i in lDecList if not i in lDelList]
print(lDecList)


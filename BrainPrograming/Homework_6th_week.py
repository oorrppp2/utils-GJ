# 1, 6, 10, 14, 16

# # 1
# lInputList = [int(x) for x in (input("주어진 리스트: ")).split()]
# lSetList = list(set(lInputList))
# print("주어진 리스트: ", lInputList)
# print("정리된 리스트: ", lSetList)

# # 6
# colors = ["red", "green", "blue"]
# values = ["#FF0000", "#008000", "#0000FF"]
# color_mapping_dict = {}
# for i, color in enumerate(colors):
#     color_mapping_dict[color] = values[i]
# print("colors =", colors)
# print("colors =", colors)
# print(color_mapping_dict)

# # 10
# set1 = {10, 20, 30, 40, 50, 60}
# set2 = {30, 40, 50, 60, 70, 80}
# print("첫 번째 세트 ", set1)
# print("두 번째 세트 ", set2)
# print("어느 한쪽에만 있는 요소들 ", (set1 | set2) - (set1 & set2))

# # 14
# sMMDDYYYY = str(input("MM/DD/YYYY: "))
# print(sMMDDYYYY, " -> ", "".join([sMMDDYYYY.split('/')[2], sMMDDYYYY.split('/')[0], sMMDDYYYY.split('/')[1]]))

# 16
import random
s = [chr(i) for i in range(33,127)]
passlen = 8
p = "".join(random.sample(s, passlen))
print("생성된 암호 =", p)


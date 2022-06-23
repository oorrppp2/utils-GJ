# import numpy as np

# week = ["일", "월", "화", "수", "목", "금", "토"]
# print("type(week): ", type(week))
# for i in range(len(week)):
#     print("week[{0}]: ".format(i) + week[i])


# today = week.index("일")
# #today = 0 # 일요일
# day_after = 100000000 # 1억

# print(week[(today + day_after) % 7] + "요일")


# print("줄바꿈은 \n 이거")


import numpy as np

a = [20, 8, 4, 2, 3, 12, 11, 16, 5, 8, 4, 17, 15, 3, 15]

first = a[0] if a[0] > a[1] else a[1]
second = a[1] if a[0] > a[1] else a[0]

for i in range(2, len(a)):
    if a[i] > first:
        second = first
        first = a[i]
    if a[i] > second and a[i] < first:
        second = a[i]

print(second)


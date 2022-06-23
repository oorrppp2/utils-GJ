
# # 1
# n = [2, 3, 5]
#
# print([i ** 2 for i in n])
# print(':'.join(map(str, n)))

# # 2
# f = ["apple", "banana", "cherry"]
# for x in f:
#     print(x, end=';')

# class R:
#     def __init__(self, width, height):
#         self.width = width
#         self.height = height
#     def area(self):
#         return self.width * self.height
# class S(R):
#     super(R)

# # 3
# def Range(start=0, end=0, unit=1):
#     if end == 0:
#         end = start
#         start = 0
#     while start < end:
#         yield start
#         start += unit
#
# print(list(Range(0, 1)))

# # 4
# def max_count(list):
#     m = 0
#     count = 0
#     for i in list:
#         m = i if i > m else m
#     for i in list:
#         if i == m:
#             count += 1
#     return count
# print(max_count([8,7,8,8,2]))

# # 5
# def my_max(*n):
#     l = n
#     m = 0
#     for i in n:
#         m = i if i > m else m
#     return m
#
# print(my_max(1,3,9,5))

# 6


# # 7
# def get_loc1():
#     return [20, 30]
# def get_loc2():
#     return {'x':20, 'y':30}
#
# loc = get_loc1()
# print('x:', loc[0], ' y:', loc[1])
# loc = get_loc2()
# print('x:', loc['x'], ' y:', loc['y'])
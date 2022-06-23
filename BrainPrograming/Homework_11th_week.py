# 8 9 11

# # 8
# def gen():
#     i = 0
#     while True:
#         i += 1
#         yield i
#
# for i in gen():
#     print(i)

# # 9
# l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# squared = list(map(lambda x: x**2, l))
# print(squared)


# import math
#
# class Circle:
#     def __init__(self, radius):
#         self.radius = radius
#
#     def __add__(self, circle):
#         return Circle(self.radius + circle.radius)
#
#     def __gt__(self, circle):
#         return self.radius > circle.radius
#
#     def __lt__(self, circle):
#         return self.radius < circle.radius
#
#     def __str__(self):
#         return "원의 반지름: " + str(self.radius)
#
#
# circle1 = Circle(5)
# print(circle1)
# circle2 = Circle(10)
# print(circle2)
#
# circle3 = circle1 + circle2
# print(circle3)
#
# if circle1 > circle2:
#     print("circle1이 circle2보다 큽니다.")
# else:
#     print("circle1이 circle2보다 작습니다.")
# if circle3 > circle2:
#     print("circle3이 circle2보다 큽니다.")
# else:
#     print("circle3이 circle2보다 작습니다.")


"""
    상속
"""

# 1 2 3

# 1
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class Point3D(Point):
    def __init__(self, x, y, z):
        super().__init__(x, y)
        self.z = z

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"

p1 = Point(10, 20)
p2 = Point3D(5, 3, 2)
print(p1)
print(p2)



# # 2
# class Address:
#     def __init__(self, street, city):
#         self.street = str(street)
#         self.city = str(city)
#
#     def __str__(self):
#         return "city: " + self.city + "\n" + "street: " + self.street
#
# class Person:
#     def __init__(self, name, email):
#         self.name = str(name)
#         self.email = str(email)
#
#     def __str__(self):
#         return "name: " + self.name + "\n" + "email: " + self.email
#
# class Contact(Address, Person):
#     def __init__(self, name, email, street, city, phone_num):
#         Person.__init__(self, name, email)
#         Address.__init__(self, street, city)
#         self.phone_num = phone_num
#
#     def __str__(self):
#         return Person.__str__(self) + "\n" + Address.__str__(self) + "\n" + "phone number: " + self.phone_num
#
# contact1 = Contact("김민수", "kim@kist.re.kr", "wolgok", "seoul", "010-0000-1111")
# contact2 = Contact("이수민", "lee@kist.re.kr", "sungbook", "seoul", "010-2222-3333")
# print(contact1)
# print(contact2)


# # 3
#
# class Function:
#     def __init__(self):
#         pass
#
#     def value(self, x):
#         pass
#
# class Quadratic(Function):
#     def __init__(self, a, b, c):
#         super().__init__()
#         self.a = a
#         self.b = b
#         self.c = c
#
#     def value(self, x):
#         return (self.a) * x ** 2 + (self.b) * x + (self.c)
#
#     def get_roots(self):
#         D = self.b ** 2 - 4*self.a*self.c
#         assert D >= 0
#
#         if D == 0:
#             sol = -self.b / (2 * self.a)
#             return sol, sol
#         else:
#             sol1 = (-self.b + pow(D, 0.5)) / (2 * self.a)
#             sol2 = (-self.b - pow(D, 0.5)) / (2 * self.a)
#             return sol1, sol2
#
#
# eq1 = Quadratic(1, 6, 9)
# eq2 = Quadratic(1, 5, 6)
# sol1, sol2 = eq1.get_roots()
# print("Solution:", sol1, sol2)
# sol1, sol2 = eq2.get_roots()
# print("Solution:", sol1, sol2)


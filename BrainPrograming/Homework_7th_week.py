"""
    객체와 클래스
"""

# 4, 6, 7

# 4
# class Rectangle():
#     def __init__(self, x, y, w, h):
#         self.x = x
#         self.y = y
#         self.width = w
#         self.height = h
#
#     def __str__(self):
#         # xmin, ymin, xmax, ymax, area
#         return f'({self.x}, {self.y}, {self.x + self.width}, {self.y + self.height}, {self.getArea()})'
#
#     def getArea(self):
#         return self.width * self.height
#
#     def overlap(self, r):
#         if (self.x <= r.x <= self.x + self.width) and \
#                 (self.y <= r.y <= self.y + self.height):
#             return True
#         else:
#             return False
#
#     def setX(self, x):
#         self.x = x
#
#     def getX(self):
#         return self.x
#
#     def setY(self, y):
#         self.y = y
#
#     def getY(self):
#         return self.y
#
#     def setWidth(self, w):
#         self.width = w
#
#     def getWidth(self):
#         return self.width
#
#     def setHeight(self, h):
#         self.height = h
#
#     def getHeight(self):
#         return self.height
#
#
# r1 = Rectangle(0, 0, 100, 100)
# r2 = Rectangle(10, 10, 100, 100)
# r3 = Rectangle(101, 90, 10, 10)
#
# if r1.overlap(r2):
#     print("r1과 r2는 서로 겹칩니다.")
# else:
#     print("r1과 r2는 서로 겹치지 않습니다.")
#
# if r1.overlap(r3):
#     print("r1과 r3는 서로 겹칩니다.")
# else:
#     print("r1과 r3는 서로 겹치지 않습니다.")

# 6
class Person():
    def __init__(self, name, mobile=None, office=None, email=None):
        self.name = name
        self.mobile = mobile
        self.office = office
        self.email = email

    def __str__(self):
        return f'(name: {self.name}, mobile: {self.mobile}, office: {self.office}, email: {self.email})'

    def setName(self, n):
        self.name = n

    def getName(self):
        return self.name

    def setMobile(self, m):
        self.mobile = m

    def getMobile(self):
        return self.name

    def setOffice(self, o):
        self.office = o

    def getOffice(self):
        return self.office

    def setEmail(self, e):
        self.email = e

    def getEmail(self):
        return self.email
#
# p1 = Person("Kim", office="123456")
# p2 = Person("Park", office="234567", mobile="010-2222-1111")
# print(p1)
# print(p2)
# p2.setEmail("park@company.com")
# print(p2)

# 7
class PhoneBook():
    def __init__(self):
        self.contacts = {}

    def __str__(self):
        return_str = ''
        for p in self.contacts:
            return_str += f'{self.contacts[p].name}\n office phone: {self.contacts[p].office}\n email address: {self.contacts[p].email}\n\n'
        return return_str

    def add(self, name, mobile=None, office=None, email=None):
        p = Person(name, mobile, office, email)
        self.contacts[name] = p


obj = PhoneBook()
obj.add("Kim", office="123456", email="kim@company.com")
obj.add("Lee", office="987656", email="lee@company.com")
print(obj)


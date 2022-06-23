# 1
def fun(n):
    sum = 0
    num = n
    if n < 0:
        n *= -1
    while n % 10 != 0:
        sum += n % 10
        n -= n % 10
        n //= 10
    if num < 0:
        return -sum
    else:
        return sum

print(fun(1234))
print(fun(-123))
print(fun(0))

# 2
# def fun(n):
#     stars = ""
#     for i in range(int(n/2)+1):
#         stars += ' '*i + '*'*n + '\n'
#         n -= 2
#     return stars
#
# print(fun(7))


# 3
# class Dot():
#     def __init__(self, n1, n2):
#         self.n1 = n1
#         self.n2 = n2
#     def __sub__(self, dot):
#         return Dot(self.n1 - dot.n1, self.n2 - dot.n2)
#     def __mul__(self, dot):
#         return Dot(self.n1 * dot.n1, self.n2 * dot.n2)
#     def __str__(self):
#         return "("+str(self.n1)+", "+str(self.n2)+")"
#
# x = Dot(2,6)
# y = Dot(5,3)
# print(x - y)
# print(x * y)

# 4
# class Ex:
#     a = 123
#     def __init__(self, x):
#         self.x = x
#
# ex = Ex(456)
# a, b = ex.a, ex.x
# print(a, b)

# 5

# from tkinter import *
#
# window = Tk()
#
# total = 100
# def button_reset_callback():
#     global total
#     total = 100
#     label['text'] = total
# def button_down_callback():
#     global total
#     total -= 1
#     label['text'] = total
#
#
# label = Label(window, width=10, height=2)
# button_reset = Button(window, text='리셋',
#         command=button_reset_callback, width=10, height=2)
# button_down = Button(window, text='카운터다운',
#         command=button_down_callback, width=10, height=2)
#
# label.grid(row=0, column=0)
# button_reset.grid(row=0, column=1)
# button_down.grid(row=1, column=0, columnspan=2)
#
# label['text'] = total
# window.mainloop()

# 6
# x = [1, 2, 3, 4]
# y = list(map(lambda x: x + 1, x))
# print(y)

# # 7
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.linspace(0, 2*np.pi, 200)
# y = np.sin(x)
#
# fig = plt.figure()
#
# ax = fig.add_subplot(2,1,1)
# ax.plot(x, y)
#
# y = np.sin(2 * x)
# ax = fig.add_subplot(2,1,2)
# ax.plot(x, y)
#
# fig.tight_layout()
# plt.show()

# # 8
# try:
#     a = open('a.txt', mode='r')
#     b = open('b.txt', mode='r')
#
#     content_ab = ""
#     content_a = a.readlines()
#     content_b = b.readlines()
#     for c in content_a:
#         content_ab += c
#     for c in content_b:
#         content_ab += c
#
#     ab = open('ab.txt', mode='w')
#     ab.write(content_ab)
#     a.close()
#     b.close()
#     ab.close()
# except FileNotFoundError as e:
#     print(e)
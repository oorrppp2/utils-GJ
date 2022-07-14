# Baekjoon sprout

from sys import stdin

# A, B = list(map(int, stdin.readline().split()))
# print(A*B)

# print("         ,r'\"7")
# print("r`-_   ,'  ,/")
# print(" \. \". L_r'")
# print("   `~\/")
# print("      |")
# print("      |")

# l = [0, 1, 2,3 ,4, 5, 6, 7]

# while l[0] != 3:
#     l.pop(0)
# print(l[0])

num = 10
next_num = int(('%04d' % num)[3]+('%04d' % num)[:3])
print(next_num)

next_num = int(('%04d' % num)[1:]+('%04d' % num)[0])
print(next_num)
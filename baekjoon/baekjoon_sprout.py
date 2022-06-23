# Baekjoon sprout

from sys import stdin

# A, B = list(map(int, stdin.readline().split()))
# print(A*B)

# p = set()
# p.add(1)
# p.add(2)
# p.add(3)
# p.add(4)

p = []
p.append(1)
p.append(2)
p.append(3)
p.append(4)

q = set()
q.add(3)
q.add(4)
q.add(5)
q.add(6)

print(p)
print(q)
print(p|list(q))
print(p)
print(q)
print(p^q)
print(p&q)
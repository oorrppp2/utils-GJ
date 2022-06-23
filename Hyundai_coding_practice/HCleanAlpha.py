import sys

# P, N = input("N K").split()
# P = int(N)
# N = int(N)
# A = list(map(int, input("A").split()))
# print(A)
# print(type(A))
import random
import time
P, N = 2, 4
# P = 100000000
# N = 1000000
M = 1000000007
A = [4, 1, 2, 3]
# A = random.sample(range(P), N)
# print(type(A))

start = time.time()
viruses = 0

for i in range(N):
    viruses = (viruses*P + A[i]%M) % M
    # viruses = (viruses + A[i]%M * pow(P, N-i-1)%M) % M

    # viruses += A[i]
    # viruses *= P
    # viruses %= 1000000007


print(viruses)
print("time :", time.time() - start)
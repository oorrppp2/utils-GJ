# Baekjoon 1645

from sys import stdin

N, K = list(map(int, stdin.readline().split()))
sticks = []
for i in range(N):
    sticks.append(int(stdin.readline()))

# if 1 in sticks:
#     print(1)
#     exit(0)
# if 2 in sticks:
#     print(2)
#     exit(0)
small = 1
large = max(sticks)+1
ans = 0
while True:
    d_stick = (small+large)//2
    if d_stick == small and ans != 0:
        break
    num_sticks = 0
    for stick in sticks:
        num_sticks += stick // d_stick

    if num_sticks >= K:
        if d_stick > ans:
            ans = d_stick
        small = d_stick
    else:
        large = d_stick
    # print("d_stick: ", d_stick, "  ,  small:, ", small, "  ,  large: ", large)

    if small >= large:
        break
print(ans)
import sys

N, K = input("N K").split()
N = int(N)
K = int(K)
line = str(input("line"))
# H : 부품
# P : 로봇
print(N)
print(K)
print(line)

# line = "HHHHHPPPPPHPHPHPHHHP"
# print(line * 1000)
# line*= 1000
# line_arr = line.split()
H = []
P = []
ans = 0
for i in range(len(line)):
    c = line[i]
    if c == 'H':
        H.append(i)
    else:
        P.append(i)

    while len(H) * len(P) > 0:
        if abs(H[0] - P[0]) <= K:
            H.remove(H[0])
            P.remove(P[0])
            ans += 1
        elif H[0] < P[0]:
            H.remove(H[0])
        else:
            P.remove(P[0])
print(ans)
#
#     if len(H) == K and len(P) == K:
#         for j in range(K):
#             if abs(H[j] - P[j]) <= K:


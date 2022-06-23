N, M = map(int, input().split())
W = [int(x) for x in input().split()]
S = [0 for i in range(10000)]
for i in range(M):
    A, B = map(int, input().split())
    if W[A-1] > W[B-1]:
        if S[A-1] == 0:
            S[A-1] = 1
        S[B-1] = -1
    elif W[A-1] < W[B-1]:
        if S[B-1] == 0:
            S[B-1] = 1
        S[A-1] = -1
print(len(list(filter(lambda x: x > 0, S))))
# S = [1, 0, 0, 0, -1, -1, 1, 1, 1, 0, 0]
# # print(len([S > 0]))
# print(list(filter(lambda x: x > 0, S)))
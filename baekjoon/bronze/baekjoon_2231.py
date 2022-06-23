Baekjoon 2231

N = input()
ans = 0
int_N = int(N)
str_N = str(N)
for i in range(int_N-(9*len(str_N)), int_N):
    if i < 1:
        continue
    if i + sum(list(map(int, str(i)))) == int_N:
        ans = i
        break

print(ans)

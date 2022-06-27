# Baekjoon 2108

from sys import stdin
N = int(stdin.readline())
l = []
for i in range(N):
    l.append(int(stdin.readline()))

if N == 1:
    print(l[0])
    print(l[0])
    print(l[0])
    print(0)
    exit(0)

l = sorted(l)
center_num = l[N//2]
freq = []
freq_n = 1
if N > 1:
    for i in range(N):

        if i == N-1:
            if l[i] != l[i-1]:
                freq.append([1, l[i]])
            else:
                freq.append([freq_n, l[i]])
        else:
            if l[i] == l[i+1]:
                freq_n += 1
            else:
                freq.append([freq_n, l[i]])
                freq_n = 1

else:
    freq = l[0]
freq = sorted(freq, reverse=True)

print(round(sum(l)/N))
print(center_num)
if freq[0][0] != freq[1][0]:
    print(freq[0][1])
else:
    for i in range(1, len(freq)-1):
        if freq[i][0] != freq[i+1][0]:
            print(freq[i-1][1])
            break
        else:
            if i == len(freq)-2:
                print(freq[i][1])
print(abs(l[-1] - l[0]))
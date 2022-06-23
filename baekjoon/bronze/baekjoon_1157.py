# Baekjoon 1157

from sys import stdin
S = stdin.readline().split()[0]
alphabet_count_dict = {}
for s in S:
    s = s.upper()
    try:
        alphabet_count_dict[s] += 1
    except:
        alphabet_count_dict[s] = 1
alphabet_count_dict = sorted(alphabet_count_dict.items(), key=lambda x:x[1])
if len(alphabet_count_dict) > 1 and alphabet_count_dict[-1][1] == alphabet_count_dict[-2][1]:
    print("?")
else:
    print(alphabet_count_dict[-1][0])
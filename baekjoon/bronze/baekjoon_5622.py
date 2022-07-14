# Baekjoon 5622

from sys import stdin

ans = 0
num_dict = {}
num_dict['A'] = num_dict['B'] = num_dict['C'] = 2
num_dict['D'] = num_dict['E'] = num_dict['F'] = 3
num_dict['G'] = num_dict['H'] = num_dict['I'] = 4
num_dict['J'] = num_dict['K'] = num_dict['L'] = 5
num_dict['M'] = num_dict['N'] = num_dict['O'] = 6
num_dict['P'] = num_dict['Q'] = num_dict['R'] = num_dict['S'] = 7
num_dict['T'] = num_dict['U'] = num_dict['V'] = 8
num_dict['W'] = num_dict['X'] = num_dict['Y'] = num_dict['Z'] = 9

dial = stdin.readline().split()[0]
for d in dial:
    ans += num_dict[d] + 1

print(ans)
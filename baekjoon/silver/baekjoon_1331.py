# Baekjoon 1331

from sys import stdin

s = stdin.readline().split()[0]
prev_s = s
ans_dict = {}
ans_dict[s] = True
start = s
s_l = []
ans = True
for i in range(35):
    s = stdin.readline().split()[0]
    s_l.append(s)

for i in range(len(s_l)):
    s = s_l[i]

    p = ord(s[0])
    q = int(s[1])
    prev_p = ord(prev_s[0])
    prev_q = int(prev_s[1])
    if i == 34:
        prev_p = ord(start[0])
        prev_q = int(start[1])
    if abs(p - prev_p) == 1:
        if abs(q - prev_q) == 2:
            prev_s = s[0] + s[1]
            try:
                if ans_dict[s]:
                    ans = False
                    break
                    # exit(0)
            except:
                ans_dict[s] = True
            continue
    elif abs(q - prev_q) == 1:
        if abs(p - prev_p) == 2:
            prev_s = s[0] + s[1]
            try:
                if ans_dict[s]:
                    ans = False
                    break
                    # exit(0)
            except:
                ans_dict[s] = True
            continue
    
    ans = False
    break

if ans:
    print("Valid")
else:
    print("Invalid")

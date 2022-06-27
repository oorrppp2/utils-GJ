# Baekjoon 4949

from sys import stdin

while True:
    s = stdin.readline()[:-1]
    if s == '.':
        break
    bracket = []
    for i in range(len(s)):
        # print(bracket)
        if s[i] == '(':
            bracket.append(')')
        elif s[i] == '[':
            bracket.append(']')
        elif s[i] == ')' or s[i] == ']':
            if len(bracket) == 0:
                bracket.append(1)
                break
            elif s[i] != bracket[-1]:
                bracket.append(1)
                break
            elif len(bracket) != 0:
                if bracket[-1] == s[i]:
                    bracket.pop()
    # print(len(s))
    print('yes' if len(bracket)==0 else 'no')
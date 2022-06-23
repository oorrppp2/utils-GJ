a=input("괄호 종류를 넣어주세요")


tmp = 0
buffer = []
if a[0] not in ['(','{','[']:
    print('False')
else:
    for i in a:
        if i in ['(','{','[']:
            if i == '(':
                buffer.append(')')
            elif i == '{':
                buffer.append('}')
            elif i == '[':
                buffer.append(']')
        else:
            if len(buffer) > 0:
                if i == buffer[-1]:
                    buffer.pop()
    print(len(buffer) == 0)

# Baekjoon 10828

from sys import stdin
N = int(stdin.readline())

stack = []
for i in range(N):
    message = stdin.readline().split()
    if message[0] == 'push':
        stack.append(message[1])
    elif message[0] == 'pop':
        if len(stack) != 0:
            print(stack.pop())
        else:
            print(-1)
    elif message[0] == 'size':
        print(len(stack))
    elif message[0] == 'empty':
        print(1 if len(stack) == 0 else 0)
    elif message[0] == 'top':
        if len(stack) != 0:
            print(stack[-1])
        else:
            print(-1)